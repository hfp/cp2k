!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Routines to reshape / redistribute tensors
!> \author Patrick Seewald
! **************************************************************************************************
MODULE dbt_reshape_ops
   #:include "dbt_macros.fypp"
   #:set maxdim = maxrank
   #:set ndims = range(2,maxdim+1)

   USE dbt_allocate_wrap, ONLY: allocate_any
   USE dbt_tas_base, ONLY: dbt_tas_copy, dbt_tas_get_info, dbt_tas_info
   USE dbt_block, ONLY: &
      dbt_iterator_type, dbt_iterator_next_block, dbt_iterator_blocks_left, dbt_iterator_start, &
      dbt_iterator_stop, dbt_get_block_data, dbt_put_block_data, dbt_reserve_blocks
   USE dbt_types, ONLY: dbt_blk_sizes, &
                        dbt_create, &
                        dbt_type, &
                        ndims_tensor, &
                        dbt_get_stored_coordinates, &
                        dbt_clear
   USE kinds, ONLY: default_string_length, dp, int_8
   USE message_passing, ONLY: &
      mp_waitall, mp_comm_type, mp_request_type

#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbt_reshape_ops'

   PUBLIC :: dbt_reshape, dbt_reshape_release

   ! Communication buffers persist across calls such that pages are faulted in once rather than per
   ! call; per block, the index holds the data offset (relative to the peer's data) and nd-index.
   REAL(dp), DIMENSION(:), ALLOCATABLE, SAVE :: data_send, data_recv
   INTEGER, DIMENSION(:), ALLOCATABLE, SAVE  :: index_send, index_recv

CONTAINS

! **************************************************************************************************
!> \brief copy data (involves reshape)
!>        tensor_out = tensor_out + tensor_in move_data memory optimization:
!>        transfer data from tensor_in to tensor_out s.t. tensor_in is empty on return
!> \author Ole Schuett
! **************************************************************************************************
   SUBROUTINE dbt_reshape(tensor_in, tensor_out, summation, move_data)

      TYPE(dbt_type), INTENT(INOUT)               :: tensor_in, tensor_out
      LOGICAL, INTENT(IN), OPTIONAL                    :: summation
      LOGICAL, INTENT(IN), OPTIONAL                    :: move_data

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbt_reshape'

      INTEGER                                            :: iproc, numnodes, nindex, &
                                                            handle, handle2, iblk, jblk, offset, ndata, &
                                                            nblks_recv_mythread
      INTEGER(KIND=int_8)                                :: ibase, idata
      INTEGER, ALLOCATABLE, DIMENSION(:, :)              :: blks_to_allocate
      TYPE(dbt_iterator_type)                            :: iter
      CHARACTER(LEN=32)                                  :: name_pack, name_unpack
      INTEGER, DIMENSION(ndims_tensor(tensor_in))        :: blk_size, ind_nd
      LOGICAL :: found, summation_prv, move_prv

      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: nblks_send_total, ndata_send_total, &
                                                            nblks_recv_total, ndata_recv_total, &
                                                            nblks_send_mythread, ndata_send_mythread
      INTEGER(KIND=int_8), ALLOCATABLE, DIMENSION(:)     :: nblks_send_displ, ndata_send_displ, &
                                                            nblks_recv_displ, ndata_recv_displ
      TYPE(mp_comm_type) :: mp_comm

      CALL timeset(routineN, handle)

      IF (PRESENT(summation)) THEN
         summation_prv = summation
      ELSE
         summation_prv = .FALSE.
      END IF

      IF (PRESENT(move_data)) THEN
         move_prv = move_data
      ELSE
         move_prv = .FALSE.
      END IF

      CPASSERT(tensor_out%valid)

      IF (.NOT. summation_prv) THEN
         CALL timeset(routineN//"_clear", handle2)
         CALL dbt_clear(tensor_out)
         CALL timestop(handle2)
      END IF

      mp_comm = tensor_in%pgrid%mp_comm_2d
      numnodes = mp_comm%num_pe
      nindex = ndims_tensor(tensor_in) + 1
      ! separate timers for blocks that need a reshape (index order differs from matrix layout)
      name_pack = routineN//"_pack"
      IF (permuted(tensor_in)) name_pack = routineN//"_pack_perm"
      name_unpack = routineN//"_unpack"
      IF (permuted(tensor_out)) name_unpack = routineN//"_unpack_perm"
      ALLOCATE (nblks_send_total(0:numnodes - 1), ndata_send_total(0:numnodes - 1), source=0)
      ALLOCATE (nblks_recv_total(0:numnodes - 1), ndata_recv_total(0:numnodes - 1), source=0)
      ALLOCATE (nblks_send_displ(0:numnodes - 1), ndata_send_displ(0:numnodes - 1))
      ALLOCATE (nblks_recv_displ(0:numnodes - 1), ndata_recv_displ(0:numnodes - 1))

!$OMP PARALLEL DEFAULT(OMP_DEFAULT_NONE_WITH_OOP) &
!$OMP SHARED(tensor_in,tensor_out,summation,mp_comm,numnodes,nindex,name_pack,name_unpack) &
!$OMP SHARED(data_send,data_recv,index_send,index_recv) &
!$OMP SHARED(nblks_send_total,ndata_send_total,nblks_recv_total,ndata_recv_total) &
!$OMP SHARED(nblks_send_displ,ndata_send_displ,nblks_recv_displ,ndata_recv_displ) &
!$OMP PRIVATE(nblks_send_mythread,ndata_send_mythread,nblks_recv_mythread) &
!$OMP PRIVATE(iter,ind_nd,blk_size,found,iproc,handle2) &
!$OMP PRIVATE(blks_to_allocate,offset,ndata,iblk,jblk,ibase,idata)
      CALL timeset(routineN//"_count", handle2)
      ALLOCATE (nblks_send_mythread(0:numnodes - 1), ndata_send_mythread(0:numnodes - 1), source=0)

      CALL dbt_iterator_start(iter, tensor_in)
      DO WHILE (dbt_iterator_blocks_left(iter))
         CALL dbt_iterator_next_block(iter, ind_nd, blk_size=blk_size)
         CALL dbt_get_stored_coordinates(tensor_out, ind_nd, iproc)
         nblks_send_mythread(iproc) = nblks_send_mythread(iproc) + 1
         ndata_send_mythread(iproc) = ndata_send_mythread(iproc) + PRODUCT(blk_size)
      END DO
      CALL dbt_iterator_stop(iter)
!$OMP CRITICAL(omp_dbt_reshape)
      nblks_send_total(:) = nblks_send_total(:) + nblks_send_mythread(:)
      ndata_send_total(:) = ndata_send_total(:) + ndata_send_mythread(:)
      nblks_send_mythread(:) = nblks_send_total(:) ! current totals indicate slot for this thread
      ndata_send_mythread(:) = ndata_send_total(:)
!$OMP END CRITICAL(omp_dbt_reshape)
!$OMP BARRIER

!$OMP MASTER
      CALL mp_comm%alltoall(nblks_send_total, nblks_recv_total, 1)
      CALL mp_comm%alltoall(ndata_send_total, ndata_recv_total, 1)
      CALL displacements(nblks_send_total, nblks_send_displ)
      CALL displacements(ndata_send_total, ndata_send_displ)
      CALL displacements(nblks_recv_total, nblks_recv_displ)
      CALL displacements(ndata_recv_total, ndata_recv_displ)
!$OMP END MASTER
!$OMP BARRIER
      CALL timestop(handle2)

      CALL timeset(TRIM(name_pack), handle2)
!$OMP MASTER
      CALL grow_index(index_send, nindex*(nblks_send_displ(numnodes - 1) + nblks_send_total(numnodes - 1)))
      CALL grow_index(index_recv, nindex*(nblks_recv_displ(numnodes - 1) + nblks_recv_total(numnodes - 1)))
      CALL grow_data(data_send, ndata_send_displ(numnodes - 1) + ndata_send_total(numnodes - 1))
      CALL grow_data(data_recv, ndata_recv_displ(numnodes - 1) + ndata_recv_total(numnodes - 1))
!$OMP END MASTER
!$OMP BARRIER

      CALL dbt_iterator_start(iter, tensor_in)
      DO WHILE (dbt_iterator_blocks_left(iter))
         CALL dbt_iterator_next_block(iter, ind_nd, blk_size=blk_size)
         CALL dbt_get_stored_coordinates(tensor_out, ind_nd, iproc)
         ! insert block data
         ndata = PRODUCT(blk_size)
         ndata_send_mythread(iproc) = ndata_send_mythread(iproc) - ndata
         offset = ndata_send_mythread(iproc)
         idata = ndata_send_displ(iproc) + offset
         CALL dbt_get_block_data(tensor_in, ind_nd, blk_size, data_send(idata + 1:idata + ndata), found)
         CPASSERT(found)
         ! insert block index
         nblks_send_mythread(iproc) = nblks_send_mythread(iproc) - 1
         ibase = nindex*(nblks_send_displ(iproc) + nblks_send_mythread(iproc))
         index_send(ibase + 1) = offset
         index_send(ibase + 2:ibase + nindex) = ind_nd(:)
      END DO
      CALL dbt_iterator_stop(iter)
      DEALLOCATE (nblks_send_mythread, ndata_send_mythread)
!$OMP BARRIER
      CALL timestop(handle2)

      CALL dbt_communicate_buffer(mp_comm, nindex, nblks_send_total, ndata_send_total, &
                                  nblks_send_displ, ndata_send_displ, nblks_recv_total, &
                                  ndata_recv_total, nblks_recv_displ, ndata_recv_displ)
!$OMP BARRIER

      CALL timeset(routineN//"_reserve", handle2)
      nblks_recv_mythread = 0
      DO iproc = 0, numnodes - 1
!$OMP DO
         DO iblk = 1, nblks_recv_total(iproc)
            nblks_recv_mythread = nblks_recv_mythread + 1
         END DO
!$OMP END DO
      END DO
      ALLOCATE (blks_to_allocate(nblks_recv_mythread, ndims_tensor(tensor_in)))

      jblk = 0
      DO iproc = 0, numnodes - 1
!$OMP DO
         DO iblk = 1, nblks_recv_total(iproc)
            jblk = jblk + 1
            ibase = nindex*(nblks_recv_displ(iproc) + iblk - 1)
            blks_to_allocate(jblk, :) = index_recv(ibase + 2:ibase + nindex)
         END DO
!$OMP END DO
      END DO
      CPASSERT(jblk == nblks_recv_mythread)
      CALL dbt_reserve_blocks(tensor_out, blks_to_allocate)
      DEALLOCATE (blks_to_allocate)
!$OMP BARRIER
      CALL timestop(handle2)

      CALL timeset(TRIM(name_unpack), handle2)
      DO iproc = 0, numnodes - 1
!$OMP DO
         DO iblk = 1, nblks_recv_total(iproc)
            ibase = nindex*(nblks_recv_displ(iproc) + iblk - 1)
            ind_nd(:) = index_recv(ibase + 2:ibase + nindex)
            CALL dbt_blk_sizes(tensor_out, ind_nd, blk_size)
            idata = ndata_recv_displ(iproc) + index_recv(ibase + 1)
            ndata = PRODUCT(blk_size)
            CALL dbt_put_block_data(tensor_out, ind_nd, blk_size, data_recv(idata + 1:idata + ndata), &
                                    summation=summation)
         END DO
!$OMP END DO
      END DO
      CALL timestop(handle2)
!$OMP END PARALLEL

      DEALLOCATE (nblks_recv_total, ndata_recv_total, nblks_recv_displ, ndata_recv_displ)
      DEALLOCATE (nblks_send_total, ndata_send_total, nblks_send_displ, ndata_send_displ)

      IF (move_prv) THEN
         CALL timeset(routineN//"_clear", handle2)
         CALL dbt_clear(tensor_in)
         CALL timestop(handle2)
      END IF

      CALL timestop(handle)
   END SUBROUTINE dbt_reshape

! **************************************************************************************************
!> \brief Release the communication buffers that persist across calls of dbt_reshape.
!> \author Hans Pabst
! **************************************************************************************************
   SUBROUTINE dbt_reshape_release()
      IF (ALLOCATED(data_send)) DEALLOCATE (data_send)
      IF (ALLOCATED(data_recv)) DEALLOCATE (data_recv)
      IF (ALLOCATED(index_send)) DEALLOCATE (index_send)
      IF (ALLOCATED(index_recv)) DEALLOCATE (index_recv)
   END SUBROUTINE dbt_reshape_release

! **************************************************************************************************
!> \brief Whether the tensor's index order differs from its matrix layout (blocks need a reshape).
!> \author Hans Pabst
! **************************************************************************************************
   FUNCTION permuted(tensor)
      TYPE(dbt_type), INTENT(IN)                            :: tensor
      LOGICAL                                               :: permuted

      INTEGER                                               :: i

      ASSOCIATE (map1_2d => tensor%nd_index_blk%map1_2d, map2_2d => tensor%nd_index_blk%map2_2d)
         permuted = .NOT. ALL([map1_2d, map2_2d] == [(i, i=1, SIZE(map1_2d) + SIZE(map2_2d))])
      END ASSOCIATE
   END FUNCTION permuted

! **************************************************************************************************
!> \brief Exclusive prefix sum of per-peer counts.
!> \author Hans Pabst
! **************************************************************************************************
   SUBROUTINE displacements(counts, displs)
      INTEGER, DIMENSION(0:), INTENT(IN)                    :: counts
      INTEGER(KIND=int_8), DIMENSION(0:), INTENT(OUT)       :: displs

      INTEGER                                               :: iproc

      displs(0) = 0
      DO iproc = 1, SIZE(counts) - 1
         displs(iproc) = displs(iproc - 1) + counts(iproc - 1)
      END DO
   END SUBROUTINE displacements

! **************************************************************************************************
!> \brief Grow a persistent buffer to hold at least the given number of elements.
!> \author Hans Pabst
! **************************************************************************************************
   SUBROUTINE grow_data(buffer, nelements)
      REAL(dp), DIMENSION(:), ALLOCATABLE, INTENT(INOUT)    :: buffer
      INTEGER(KIND=int_8), INTENT(IN)                       :: nelements

      IF (ALLOCATED(buffer)) THEN
         IF (SIZE(buffer, KIND=int_8) < nelements) DEALLOCATE (buffer)
      END IF
      IF (.NOT. ALLOCATED(buffer)) ALLOCATE (buffer(MAX(nelements + nelements/4, 1_int_8)))
   END SUBROUTINE grow_data

! **************************************************************************************************
!> \brief Grow a persistent buffer to hold at least the given number of elements.
!> \author Hans Pabst
! **************************************************************************************************
   SUBROUTINE grow_index(buffer, nelements)
      INTEGER, DIMENSION(:), ALLOCATABLE, INTENT(INOUT)     :: buffer
      INTEGER(KIND=int_8), INTENT(IN)                       :: nelements

      IF (ALLOCATED(buffer)) THEN
         IF (SIZE(buffer, KIND=int_8) < nelements) DEALLOCATE (buffer)
      END IF
      IF (.NOT. ALLOCATED(buffer)) ALLOCATE (buffer(MAX(nelements + nelements/4, 1_int_8)))
   END SUBROUTINE grow_index

! **************************************************************************************************
!> \brief communicate buffer
!> \author Patrick Seewald
! **************************************************************************************************
   SUBROUTINE dbt_communicate_buffer(mp_comm, nindex, nblks_send, ndata_send, nblks_send_displ, &
                                     ndata_send_displ, nblks_recv, ndata_recv, nblks_recv_displ, &
                                     ndata_recv_displ)
      TYPE(mp_comm_type), INTENT(IN)                        :: mp_comm
      INTEGER, INTENT(IN)                                   :: nindex
      INTEGER, DIMENSION(0:), INTENT(IN)                    :: nblks_send, ndata_send, &
                                                               nblks_recv, ndata_recv
      INTEGER(KIND=int_8), DIMENSION(0:), INTENT(IN)        :: nblks_send_displ, ndata_send_displ, &
                                                               nblks_recv_displ, ndata_recv_displ

      CHARACTER(LEN=*), PARAMETER :: routineN = 'dbt_communicate_buffer'

      INTEGER                                               :: iproc, numnodes, &
                                                               rec_counter, send_counter
      INTEGER(KIND=int_8)                                   :: i, i0, i1, d0, d1
      TYPE(mp_request_type), ALLOCATABLE, DIMENSION(:, :)   :: req_array
      INTEGER                                               :: handle

      CALL timeset(routineN, handle)
      numnodes = mp_comm%num_pe

      IF (numnodes > 1) THEN
!$OMP MASTER
         send_counter = 0
         rec_counter = 0

         ALLOCATE (req_array(1:numnodes, 4))

         DO iproc = 0, numnodes - 1
            IF (nblks_recv(iproc) > 0) THEN
               rec_counter = rec_counter + 1
               i0 = nindex*nblks_recv_displ(iproc) + 1
               i1 = nindex*(nblks_recv_displ(iproc) + nblks_recv(iproc))
               d0 = ndata_recv_displ(iproc) + 1
               d1 = ndata_recv_displ(iproc) + ndata_recv(iproc)
               CALL mp_comm%irecv(index_recv(i0:i1), iproc, req_array(rec_counter, 3), tag=4)
               CALL mp_comm%irecv(data_recv(d0:d1), iproc, req_array(rec_counter, 4), tag=7)
            END IF
         END DO

         DO iproc = 0, numnodes - 1
            IF (nblks_send(iproc) > 0) THEN
               send_counter = send_counter + 1
               i0 = nindex*nblks_send_displ(iproc) + 1
               i1 = nindex*(nblks_send_displ(iproc) + nblks_send(iproc))
               d0 = ndata_send_displ(iproc) + 1
               d1 = ndata_send_displ(iproc) + ndata_send(iproc)
               CALL mp_comm%isend(index_send(i0:i1), iproc, req_array(send_counter, 1), tag=4)
               CALL mp_comm%isend(data_send(d0:d1), iproc, req_array(send_counter, 2), tag=7)
            END IF
         END DO

         IF (send_counter > 0) THEN
            CALL mp_waitall(req_array(1:send_counter, 1:2))
         END IF
         IF (rec_counter > 0) THEN
            CALL mp_waitall(req_array(1:rec_counter, 3:4))
         END IF
!$OMP END MASTER

      ELSE
         i1 = nindex*INT(nblks_send(0), int_8)
         d1 = INT(ndata_send(0), int_8)
!$OMP DO SCHEDULE(static)
         DO i = 1, i1
            index_recv(i) = index_send(i)
         END DO
!$OMP END DO NOWAIT
!$OMP DO SCHEDULE(static)
         DO i = 1, d1
            data_recv(i) = data_send(i)
         END DO
!$OMP END DO
      END IF
      CALL timestop(handle)

   END SUBROUTINE dbt_communicate_buffer

END MODULE dbt_reshape_ops
