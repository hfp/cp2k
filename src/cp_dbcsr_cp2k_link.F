!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief  Routines that link DBCSR and CP2K concepts together
!> \author Ole Schuett
!> \par History
!>         01.2014 created
! **************************************************************************************************
MODULE cp_dbcsr_cp2k_link
   USE ao_util,                         ONLY: exp_radius
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE basis_set_types,                 ONLY: gto_basis_set_p_type,&
                                              gto_basis_set_type
   USE bibliography,                    ONLY: Borstnik2014,&
                                              Heinecke2016,&
                                              Schuett2016,&
                                              cite_reference
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_finalize, dbcsr_get_block_p, dbcsr_get_default_config, dbcsr_get_matrix_type, &
        dbcsr_has_symmetry, dbcsr_reserve_blocks, dbcsr_set, dbcsr_set_config, dbcsr_type, &
        dbcsr_type_no_symmetry
   USE cp_dbcsr_operations,             ONLY: max_elements_per_block
   USE input_keyword_types,             ONLY: keyword_create,&
                                              keyword_release,&
                                              keyword_type
   USE input_section_types,             ONLY: section_add_keyword,&
                                              section_add_subsection,&
                                              section_create,&
                                              section_release,&
                                              section_type,&
                                              section_vals_get_subs_vals,&
                                              section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: dp,&
                                              real_8
   USE orbital_pointers,                ONLY: nso
   USE qs_integral_utils,               ONLY: basis_set_list_setup
   USE qs_kind_types,                   ONLY: qs_kind_type
   USE qs_ks_types,                     ONLY: get_ks_env,&
                                              qs_ks_env_type
   USE qs_neighbor_list_types,          ONLY: get_iterator_info,&
                                              get_neighbor_list_set_p,&
                                              neighbor_list_iterate,&
                                              neighbor_list_iterator_create,&
                                              neighbor_list_iterator_p_type,&
                                              neighbor_list_iterator_release,&
                                              neighbor_list_set_p_type
   USE string_utilities,                ONLY: s2a
#include "./base/base_uses.f90"

   IMPLICIT NONE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'cp_dbcsr_cp2k_link'

   PUBLIC :: create_dbcsr_section
   PUBLIC :: cp_dbcsr_config
   PUBLIC :: cp_dbcsr_alloc_block_from_nbl
   PUBLIC :: cp_dbcsr_to_csr_screening

   PRIVATE

   ! Possible drivers to use for matrix multiplications
   INTEGER, PARAMETER :: mm_driver_auto = 0
   INTEGER, PARAMETER :: mm_driver_matmul = 1
   INTEGER, PARAMETER :: mm_driver_blas = 2
   INTEGER, PARAMETER :: mm_driver_smm = 3
   INTEGER, PARAMETER :: mm_driver_xsmm = 4

   CHARACTER(len=*), PARAMETER :: mm_name_auto = "AUTO", &
                                  mm_name_blas = "BLAS", &
                                  mm_name_matmul = "MATMUL", &
                                  mm_name_smm = "SMM", &
                                  mm_name_xsmm = "XSMM"
CONTAINS

! **************************************************************************************************
!> \brief   Creates the dbcsr section for configuring DBCSR
!> \param section ...
!> \date    2011-04-05
!> \author  Urban Borstnik
! **************************************************************************************************
   SUBROUTINE create_dbcsr_section(section)
      TYPE(section_type), POINTER                        :: section

      INTEGER                                            :: idefault
      LOGICAL                                            :: ldefault
      REAL(KIND=dp)                                      :: rdefault
      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: subsection

      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="DBCSR", &
                          description="Configuration options for the DBCSR library.", &
                          n_keywords=1, n_subsections=0, repeats=.FALSE., &
                          citations=(/Borstnik2014, Schuett2016/))

      NULLIFY (keyword)

      CALL keyword_create(keyword, __LOCATION__, name="mm_stack_size", &
                          description="Size of multiplication parameter stack." &
                          //" A negative value leaves the decision up to DBCSR.", &
                          usage="mm_stack_size 1000", &
                          default_i_val=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="mm_driver", &
                          description="Select which backend to use preferably "// &
                          "for matrix block multiplications on the host.", &
                          usage="mm_driver blas", &
                          default_i_val=mm_driver_auto, &
                          enum_c_vals=s2a("AUTO", "BLAS", "MATMUL", "SMM", "XSMM"), &
                          enum_i_vals=(/mm_driver_auto, mm_driver_blas, mm_driver_matmul, mm_driver_smm, &
                                        mm_driver_xsmm/), &
                          enum_desc=s2a("Choose automatically the best available driver", &
                                        "BLAS (requires the BLAS library at link time)", &
                                        "Fortran MATMUL", &
                                        "Library optimised for Small Matrix Multiplies "// &
                                        "(requires the SMM library at link time)", &
                                        "LIBXSMM"), &
                          citations=(/Heinecke2016/))
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(avg_elements_images=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="avg_elements_images", &
                          description="Average number of elements (dense limit)" &
                          //" for each image, which also corresponds to" &
                          //" the average number of elements exchanged between MPI processes" &
                          //" during the operations." &
                          //" A negative or zero value means unlimited.", &
                          usage="avg_elements_images 10000", &
                          default_i_val=idefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(num_mult_images=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="num_mult_images", &
                          description="Multiplicative factor for number of virtual images.", &
                          usage="num_mult_images 2", &
                          default_i_val=idefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(use_mpi_allocator=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="use_mpi_allocator", &
                          description="Use MPI allocator" &
                          //" to allocate buffers used in MPI communications.", &
                          usage="use_mpi_allocator T", &
                          default_l_val=ldefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(use_mpi_rma=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="use_mpi_rma", &
                          description="Use RMA for MPI communications" &
                          //" for each image, which also corresponds to" &
                          //" the number of elements exchanged between MPI processes" &
                          //" during the operations.", &
                          usage="use_mpi_rma F", &
                          default_l_val=ldefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(num_layers_3D=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="num_layers_3D", &
                          description="Number of layers for the 3D multplication algorithm.", &
                          usage="num_layers_3D 1", &
                          default_i_val=idefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(nstacks=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="n_size_mnk_stacks", &
                          description="Number of stacks to use for distinct atomic sizes" &
                          //" (e.g., 2 for a system of mostly waters). ", &
                          usage="n_size_mnk_stacks 2", &
                          default_i_val=idefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(use_comm_thread=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="use_comm_thread", &
                          description="During multiplication, use a thread to periodically poll" &
                          //" MPI to progress outstanding message completions.  This is" &
                          //" beneficial on systems without a DMA-capable network adapter" &
                          //" e.g. Cray XE6.", &
                          usage="use_comm_thread T", &
                          default_l_val=ldefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="MAX_ELEMENTS_PER_BLOCK", &
                          description="Default block size for turning dense matrices in blocked ones", &
                          usage="MAX_ELEMENTS_PER_BLOCK 32", &
                          default_i_val=max_elements_per_block)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="comm_thread_load", &
                          description="If a communications thread is used, specify how much " &
                          //"multiplication workload (%) the thread should perform in " &
                          //"addition to communication tasks. " &
                          //"A negative value leaves the decision up to DBCSR.", &
                          usage="comm_thread_load 50", &
                          default_i_val=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(multrec_limit=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="multrec_limit", &
                          description="Recursion limit of cache oblivious multrec algorithm.", &
                          default_i_val=idefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(use_mempools_cpu=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="use_mempools_cpu", &
                          description="Enable memory pools on the CPU.", &
                          default_l_val=ldefault)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      NULLIFY (subsection)
      CALL section_create(subsection, __LOCATION__, name="TENSOR", &
                          description="Configuration options for Tensors.", &
                          n_keywords=1, n_subsections=0, repeats=.FALSE.)

      CALL dbcsr_get_default_config(tas_split_factor=rdefault)
      CALL keyword_create(keyword, __LOCATION__, name="TAS_SPLIT_FACTOR", &
                          description="Parameter for hybrid DBCSR-TAS matrix-matrix multiplication algorithm: "// &
                          "a TAS matrix is split into s submatrices with s = N_max/(N_min*f) with f "// &
                          "given by this parameter and N_max/N_min the max/min occupancies of the matrices "// &
                          "involved in a multiplication. A large value makes the multiplication Cannon-based "// &
                          "(s=1) and a small value (> 0) makes the multiplication based on TAS algorithm "// &
                          "(s=number of MPI ranks)", &
                          default_r_val=rdefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

      !---------------------------------------------------------------------------
      NULLIFY (subsection)
      CALL section_create(subsection, __LOCATION__, name="ACC", &
                          description="Configuration options for the ACC-Driver.", &
                          n_keywords=1, n_subsections=0, repeats=.FALSE.)

      CALL dbcsr_get_default_config(accdrv_thread_buffers=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="thread_buffers", &
                          description="Number of transfer-buffers associated with each thread and corresponding stream.", &
                          default_i_val=idefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_avoid_after_busy=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="avoid_after_busy", &
                          description="If enabled, stacks are not processed by the acc-driver " &
                          //"after it has signaled congestion during a round of flushing. " &
                          //"For the next round of flusing the driver is used again.", &
                          default_l_val=ldefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_min_flop_process=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="min_flop_process", &
                          description="Only process stacks with more than the given number of " &
                          //"floating-point operations per stack-entry (2*m*n*k).", &
                          default_i_val=idefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_stack_sort=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="stack_sort", &
                          description="Sort multiplication stacks according to C-access.", &
                          default_l_val=ldefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_min_flop_sort=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="min_flop_sort", &
                          description="Only sort stacks with more than the given number of " &
                          //"floating-point operations per stack-entry (2*m*n*k). " &
                          //"Alternatively, the stacks are roughly ordered through a " &
                          //"binning-scheme by Peter Messmer. (Depends on ACC%STACK_SORT)", &
                          default_i_val=idefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_do_inhomogenous=ldefault)
      CALL keyword_create(keyword, __LOCATION__, name="process_inhomogenous", &
                          description="If enabled, inhomogenous stacks are also processed by the acc driver.", &
                          default_l_val=ldefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_binning_nbins=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="binning_nbins", &
                          description="Number of bins used when ordering " &
                          //"the stacks with the binning-scheme.", &
                          default_i_val=idefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL dbcsr_get_default_config(accdrv_binning_binsize=idefault)
      CALL keyword_create(keyword, __LOCATION__, name="binning_binsize", &
                          description="Size of bins used when ordering " &
                          //"the stacks with the binning-scheme.", &
                          default_i_val=idefault)
      CALL section_add_keyword(subsection, keyword)
      CALL keyword_release(keyword)

      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

   END SUBROUTINE create_dbcsr_section

! **************************************************************************************************
!> \brief Configures options for DBCSR
!> \param root_section ...
! **************************************************************************************************
   SUBROUTINE cp_dbcsr_config(root_section)
      TYPE(section_vals_type), POINTER                   :: root_section

      CHARACTER(len=*), PARAMETER                        :: routineN = 'cp_dbcsr_config'

      INTEGER                                            :: handle, ival
      LOGICAL                                            :: lval
      REAL(kind=real_8)                                  :: rval
      TYPE(section_vals_type), POINTER                   :: dbcsr_section

      CALL timeset(routineN, handle)

      CALL cite_reference(Borstnik2014)
      CALL cite_reference(Schuett2016)

      dbcsr_section => section_vals_get_subs_vals(root_section, "GLOBAL%DBCSR")

      CALL section_vals_val_get(dbcsr_section, "mm_stack_size", i_val=ival)
      CALL dbcsr_set_config(mm_stack_size=ival)

      CALL section_vals_val_get(dbcsr_section, "MAX_ELEMENTS_PER_BLOCK", i_val=max_elements_per_block)

      CALL section_vals_val_get(dbcsr_section, "avg_elements_images", i_val=ival)
      CALL dbcsr_set_config(avg_elements_images=ival)

      CALL section_vals_val_get(dbcsr_section, "num_mult_images", i_val=ival)
      CALL dbcsr_set_config(num_mult_images=ival)

      CALL section_vals_val_get(dbcsr_section, "n_size_mnk_stacks", i_val=ival)
      CALL dbcsr_set_config(nstacks=ival)

      CALL section_vals_val_get(dbcsr_section, "use_mpi_allocator", l_val=lval)
      CALL dbcsr_set_config(use_mpi_allocator=lval)

      CALL section_vals_val_get(dbcsr_section, "use_mpi_rma", l_val=lval)
      CALL dbcsr_set_config(use_mpi_rma=lval)

      CALL section_vals_val_get(dbcsr_section, "num_layers_3D", i_val=ival)
      CALL dbcsr_set_config(num_layers_3D=ival)

      CALL section_vals_val_get(dbcsr_section, "use_comm_thread", l_val=lval)
      CALL dbcsr_set_config(use_comm_thread=lval)

      CALL section_vals_val_get(dbcsr_section, "comm_thread_load", i_val=ival)
      CALL dbcsr_set_config(comm_thread_load=ival)

      CALL section_vals_val_get(dbcsr_section, "multrec_limit", i_val=ival)
      CALL dbcsr_set_config(multrec_limit=ival)

      CALL section_vals_val_get(dbcsr_section, "use_mempools_cpu", l_val=lval)
      CALL dbcsr_set_config(use_mempools_cpu=lval)

      CALL section_vals_val_get(dbcsr_section, "TENSOR%tas_split_factor", r_val=rval)
      CALL dbcsr_set_config(tas_split_factor=rval)

      CALL section_vals_val_get(dbcsr_section, "ACC%thread_buffers", i_val=ival)
      CALL dbcsr_set_config(accdrv_thread_buffers=ival)

      CALL section_vals_val_get(dbcsr_section, "ACC%min_flop_process", i_val=ival)
      CALL dbcsr_set_config(accdrv_min_flop_process=ival)

      CALL section_vals_val_get(dbcsr_section, "ACC%stack_sort", l_val=lval)
      CALL dbcsr_set_config(accdrv_stack_sort=lval)

      CALL section_vals_val_get(dbcsr_section, "ACC%min_flop_sort", i_val=ival)
      CALL dbcsr_set_config(accdrv_min_flop_sort=ival)

      CALL section_vals_val_get(dbcsr_section, "ACC%process_inhomogenous", l_val=lval)
      CALL dbcsr_set_config(accdrv_do_inhomogenous=lval)

      CALL section_vals_val_get(dbcsr_section, "ACC%avoid_after_busy", l_val=lval)
      CALL dbcsr_set_config(accdrv_avoid_after_busy=lval)

      CALL section_vals_val_get(dbcsr_section, "ACC%binning_nbins", i_val=ival)
      CALL dbcsr_set_config(accdrv_binning_nbins=ival)

      CALL section_vals_val_get(dbcsr_section, "ACC%binning_binsize", i_val=ival)
      CALL dbcsr_set_config(accdrv_binning_binsize=ival)

      CALL section_vals_val_get(dbcsr_section, "mm_driver", i_val=ival)
      SELECT CASE (ival)
      CASE (mm_driver_auto)
         CALL dbcsr_set_config(mm_driver="AUTO")
#if defined(__LIBXSMM)
         CALL cite_reference(Heinecke2016)
#endif
      CASE (mm_driver_blas)
         CALL dbcsr_set_config(mm_driver="BLAS")
      CASE (mm_driver_matmul)
         CALL dbcsr_set_config(mm_driver="MATMUL")
      CASE (mm_driver_smm)
         CALL dbcsr_set_config(mm_driver="SMM")
      CASE (mm_driver_xsmm)
         CALL dbcsr_set_config(mm_driver="XSMM")
         CALL cite_reference(Heinecke2016)
      CASE DEFAULT
         CPABORT("Unknown mm_driver")
      END SELECT

      CALL timestop(handle)
   END SUBROUTINE cp_dbcsr_config

! **************************************************************************************************
!> \brief allocate the blocks of a dbcsr based on the neighbor list
!> \param matrix        the matrix
!> \param sab_orb       the corresponding neighbor list
!> \param desymmetrize  Allocate all block of a non-symmetric matrix from a symmetric list
!> \par History
!>      11.2009 created vw
!>      01.2014 moved here from cp_dbcsr_operations (Ole Schuett)
!> \author vw
!> \note
! **************************************************************************************************

   SUBROUTINE cp_dbcsr_alloc_block_from_nbl(matrix, sab_orb, desymmetrize)

      TYPE(dbcsr_type)                                   :: matrix
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_orb
      LOGICAL, INTENT(IN), OPTIONAL                      :: desymmetrize

      CHARACTER(LEN=*), PARAMETER :: routineN = 'cp_dbcsr_alloc_block_from_nbl'

      CHARACTER(LEN=1)                                   :: symmetry
      INTEGER                                            :: blk_cnt, handle, iatom, icol, inode, &
                                                            irow, jatom, last_jatom, nadd
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: cols, rows, tmp
      LOGICAL                                            :: alloc_full, is_symmetric, new_atom_b
      TYPE(neighbor_list_iterator_p_type), &
         DIMENSION(:), POINTER                           :: nl_iterator

      CALL timeset(routineN, handle)

      symmetry = dbcsr_get_matrix_type(matrix)

      CPASSERT(ASSOCIATED(sab_orb))

      CALL get_neighbor_list_set_p(neighbor_list_sets=sab_orb, symmetric=is_symmetric)
      alloc_full = .FALSE.
      IF (PRESENT(desymmetrize)) THEN
         IF (desymmetrize .AND. (symmetry == dbcsr_type_no_symmetry)) THEN
            IF (is_symmetric) alloc_full = .TRUE.
         END IF
      END IF

      CALL dbcsr_finalize(matrix)
      ALLOCATE (rows(3), cols(3))
      blk_cnt = 0
      nadd = 1
      IF (alloc_full) nadd = 2

      CALL neighbor_list_iterator_create(nl_iterator, sab_orb)
      DO WHILE (neighbor_list_iterate(nl_iterator) == 0)
         CALL get_iterator_info(nl_iterator, iatom=iatom, jatom=jatom, inode=inode)
         IF (inode == 1) last_jatom = 0
         IF (jatom /= last_jatom) THEN
            new_atom_b = .TRUE.
            last_jatom = jatom
         ELSE
            new_atom_b = .FALSE.
            CYCLE
         END IF
         IF (blk_cnt + nadd .GT. SIZE(rows)) THEN
            ALLOCATE (tmp(blk_cnt + nadd))
            tmp(1:blk_cnt) = rows(1:blk_cnt)
            DEALLOCATE (rows)
            ALLOCATE (rows((blk_cnt + nadd)*2))
            rows(1:blk_cnt) = tmp(1:blk_cnt)
            tmp(1:blk_cnt) = cols(1:blk_cnt)
            DEALLOCATE (cols)
            ALLOCATE (cols((blk_cnt + nadd)*2))
            cols(1:blk_cnt) = tmp(1:blk_cnt)
            DEALLOCATE (tmp)
         END IF
         IF (alloc_full) THEN
            blk_cnt = blk_cnt + 1
            rows(blk_cnt) = iatom
            cols(blk_cnt) = jatom
            IF (iatom /= jatom) THEN
               blk_cnt = blk_cnt + 1
               rows(blk_cnt) = jatom
               cols(blk_cnt) = iatom
            END IF
         ELSE
            blk_cnt = blk_cnt + 1
            IF (symmetry == dbcsr_type_no_symmetry) THEN
               rows(blk_cnt) = iatom
               cols(blk_cnt) = jatom
            ELSE
               IF (iatom <= jatom) THEN
                  irow = iatom
                  icol = jatom
               ELSE
                  irow = jatom
                  icol = iatom
               END IF
               rows(blk_cnt) = irow
               cols(blk_cnt) = icol
            END IF
         END IF

      END DO
      CALL neighbor_list_iterator_release(nl_iterator)

      !
      CALL dbcsr_reserve_blocks(matrix, rows(1:blk_cnt), cols(1:blk_cnt))
      DEALLOCATE (rows)
      DEALLOCATE (cols)
      CALL dbcsr_finalize(matrix)

      CALL timestop(handle)

   END SUBROUTINE cp_dbcsr_alloc_block_from_nbl

! **************************************************************************************************
!> \brief Apply distance screening to refine sparsity pattern of matrices in CSR
!>        format (using eps_pgf_orb). Currently this is used for the external
!>        library PEXSI.
!> \param ks_env ...
!> \param[in, out] csr_sparsity DBCSR matrix defining CSR sparsity pattern.
!>                              This matrix must be initialized and allocated
!>                              with exactly the same DBCSR sparsity pattern as
!>                              the DBCSR matrix that is used to create the CSR
!>                              matrix. It must have symmetric DBCSR format and
!>                              must not be filtered.
!> \par History
!>      02.2015 created [Patrick Seewald]
!> \author Patrick Seewald
! **************************************************************************************************
   SUBROUTINE cp_dbcsr_to_csr_screening(ks_env, csr_sparsity)
      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_type), INTENT(INOUT)                    :: csr_sparsity

      CHARACTER(len=*), PARAMETER :: routineN = 'cp_dbcsr_to_csr_screening'

      INTEGER :: atom_a, atom_b, handle, iatom, icol, ikind, ipgf, irow, iset, isgf, ishell, &
         jatom, jkind, jpgf, jset, jsgf, jshell, nkind, nset_a, nset_b
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: atom_of_kind
      INTEGER, DIMENSION(:), POINTER                     :: npgf_a, npgf_b, nshell_a, nshell_b
      INTEGER, DIMENSION(:, :), POINTER                  :: l_a, l_b
      LOGICAL                                            :: do_symmetric, found
      REAL(KIND=dp)                                      :: dab, eps_pgf_orb, r_a, r_b
      REAL(KIND=dp), DIMENSION(3)                        :: rab
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: rpgfa, rpgfb, zet_a, zet_b
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: gcc_a, gcc_b
      REAL(KIND=real_8), DIMENSION(:, :), POINTER        :: screen_blk
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list_a, basis_set_list_b
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(neighbor_list_iterator_p_type), &
         DIMENSION(:), POINTER                           :: nl_iterator
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: neighbour_list
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      NULLIFY (screen_blk, atomic_kind_set, basis_set_list_a, &
               basis_set_list_b, basis_set_a, basis_set_b, nl_iterator, &
               qs_kind_set, dft_control)

      CALL timeset(routineN, handle)

      CPASSERT(dbcsr_has_symmetry(csr_sparsity))

      CALL get_ks_env(ks_env, &
                      sab_orb=neighbour_list, &
                      atomic_kind_set=atomic_kind_set, &
                      qs_kind_set=qs_kind_set, &
                      dft_control=dft_control)

      eps_pgf_orb = dft_control%qs_control%eps_pgf_orb

      nkind = SIZE(qs_kind_set)
      CALL get_atomic_kind_set(atomic_kind_set, atom_of_kind=atom_of_kind)
      CPASSERT(SIZE(neighbour_list) > 0)
      CALL get_neighbor_list_set_p(neighbor_list_sets=neighbour_list, symmetric=do_symmetric)
      CPASSERT(do_symmetric)
      ALLOCATE (basis_set_list_a(nkind), basis_set_list_b(nkind))
      CALL basis_set_list_setup(basis_set_list_a, "ORB", qs_kind_set)
      CALL basis_set_list_setup(basis_set_list_b, "ORB", qs_kind_set)

      ! csr_sparsity can obtain values 0 (if zero element) or 1 (if non-zero element)
      CALL dbcsr_set(csr_sparsity, 0.0_dp)

      CALL neighbor_list_iterator_create(nl_iterator, neighbour_list)

      ! Iterate over interacting pairs of atoms corresponding to non-zero
      ! DBCSR blocks
      DO WHILE (neighbor_list_iterate(nl_iterator) == 0)
         CALL get_iterator_info(nl_iterator, &
                                ikind=ikind, jkind=jkind, &
                                iatom=iatom, jatom=jatom, &
                                r=rab)

         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE

         atom_a = atom_of_kind(iatom)
         atom_b = atom_of_kind(jatom)

         nset_a = basis_set_a%nset
         nset_b = basis_set_b%nset
         npgf_a => basis_set_a%npgf
         npgf_b => basis_set_b%npgf
         nshell_a => basis_set_a%nshell
         nshell_b => basis_set_b%nshell

         l_a => basis_set_a%l
         l_b => basis_set_b%l
         gcc_a => basis_set_a%gcc
         gcc_b => basis_set_b%gcc
         zet_a => basis_set_a%zet
         zet_b => basis_set_b%zet

         rpgfa => basis_set_a%pgf_radius
         rpgfb => basis_set_b%pgf_radius

         IF (iatom <= jatom) THEN
            irow = iatom
            icol = jatom
         ELSE
            irow = jatom
            icol = iatom
         END IF

         CALL dbcsr_get_block_p(matrix=csr_sparsity, row=irow, col=icol, &
                                block=screen_blk, found=found)

         CPASSERT(found)

         ! Distance between atoms a and b
         dab = SQRT(rab(1)**2 + rab(2)**2 + rab(3)**2)

         ! iterate over pairs of primitive GTOs i,j, get their radii r_i, r_j according
         ! to eps_pgf_orb. Define all matrix elements as non-zero to which a
         ! contribution from two Gaussians i,j exists with r_i + r_j >= dab.

         isgf = 0
         DO iset = 1, nset_a
            DO ishell = 1, nshell_a(iset)
               jsgf = 0
               DO jset = 1, nset_b
                  DO jshell = 1, nshell_b(jset)
                     gto_loop: DO ipgf = 1, npgf_a(iset)
                        DO jpgf = 1, npgf_b(jset)
                           IF (rpgfa(ipgf, iset) + rpgfb(jpgf, jset) .GE. dab) THEN
                              ! more selective screening with radius calculated for each primitive GTO
                              r_a = exp_radius(l_a(ishell, iset), &
                                               zet_a(ipgf, iset), &
                                               eps_pgf_orb, &
                                               gcc_a(ipgf, ishell, iset))
                              r_b = exp_radius(l_b(jshell, jset), &
                                               zet_b(jpgf, jset), &
                                               eps_pgf_orb, &
                                               gcc_b(jpgf, jshell, jset))
                              IF (r_a + r_b .GE. dab) THEN
                                 IF (irow .EQ. iatom) THEN
                                    screen_blk(isgf + 1:isgf + nso(l_a(ishell, iset)), &
                                               jsgf + 1:jsgf + nso(l_b(jshell, jset))) = 1.0_dp
                                 ELSE
                                    screen_blk(jsgf + 1:jsgf + nso(l_b(jshell, jset)), &
                                               isgf + 1:isgf + nso(l_a(ishell, iset))) = 1.0_dp
                                 END IF
                                 EXIT gto_loop
                              END IF
                           END IF
                        END DO
                     END DO gto_loop
                     jsgf = jsgf + nso(l_b(jshell, jset))
                  END DO
               END DO
               isgf = isgf + nso(l_a(ishell, iset))
            END DO
         END DO
      END DO

      CALL neighbor_list_iterator_release(nl_iterator)
      DEALLOCATE (basis_set_list_a, basis_set_list_b)

      CALL timestop(handle)
   END SUBROUTINE cp_dbcsr_to_csr_screening

END MODULE cp_dbcsr_cp2k_link
