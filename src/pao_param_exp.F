!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Original matrix exponential parametrization
!> \author Ole Schuett
! **************************************************************************************************
MODULE pao_param_exp
   USE basis_set_types,                 ONLY: gto_basis_set_type
   USE cp_dbcsr_api,                    ONLY: &
        dbcsr_create, dbcsr_get_block_p, dbcsr_get_info, dbcsr_iterator_blocks_left, &
        dbcsr_iterator_next_block, dbcsr_iterator_start, dbcsr_iterator_stop, dbcsr_iterator_type, &
        dbcsr_p_type, dbcsr_release, dbcsr_set, dbcsr_type
   USE cp_dbcsr_contrib,                ONLY: dbcsr_reserve_diag_blocks
   USE dm_ls_scf_types,                 ONLY: ls_scf_env_type
   USE kinds,                           ONLY: dp
   USE mathlib,                         ONLY: diag_antisym,&
                                              diamat_all
   USE pao_param_methods,               ONLY: pao_calc_AB_from_U,&
                                              pao_calc_grad_lnv_wrt_U
   USE pao_potentials,                  ONLY: pao_guess_initial_potential
   USE pao_types,                       ONLY: pao_env_type
   USE qs_environment_types,            ONLY: get_qs_env,&
                                              qs_environment_type
   USE qs_kind_types,                   ONLY: get_qs_kind,&
                                              qs_kind_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'pao_param_exp'

   PUBLIC :: pao_param_init_exp, pao_param_finalize_exp, pao_calc_AB_exp
   PUBLIC :: pao_param_count_exp, pao_param_initguess_exp

CONTAINS

! **************************************************************************************************
!> \brief Initialize matrix exponential parametrization
!> \param pao ...
!> \param qs_env ...
! **************************************************************************************************
   SUBROUTINE pao_param_init_exp(pao, qs_env)
      TYPE(pao_env_type), POINTER                        :: pao
      TYPE(qs_environment_type), POINTER                 :: qs_env

      CHARACTER(len=*), PARAMETER :: routineN = 'pao_param_init_exp'

      INTEGER                                            :: acol, arow, handle, iatom, N
      LOGICAL                                            :: found
      REAL(dp), DIMENSION(:), POINTER                    :: H_evals
      REAL(dp), DIMENSION(:, :), POINTER                 :: block_H, block_H0, block_N, block_U0, &
                                                            block_V0, H_evecs
      TYPE(dbcsr_iterator_type)                          :: iter
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s

      CALL timeset(routineN, handle)

      CALL get_qs_env(qs_env, matrix_s=matrix_s)

      ! allocate matrix_U0
      CALL dbcsr_create(pao%matrix_U0, &
                        name="PAO matrix_U0", &
                        matrix_type="N", &
                        dist=pao%diag_distribution, &
                        template=matrix_s(1)%matrix)
      CALL dbcsr_reserve_diag_blocks(pao%matrix_U0)

      ! diagonalize each block of H0 and store eigenvectors in U0
!$OMP PARALLEL DEFAULT(NONE) SHARED(pao,qs_env) &
!$OMP PRIVATE(iter,arow,acol,iatom,N,found,block_H0,block_V0,block_N,block_H,block_U0,H_evecs,H_evals)
      CALL dbcsr_iterator_start(iter, pao%matrix_U0)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, arow, acol, block_U0)
         iatom = arow; CPASSERT(arow == acol)
         CALL dbcsr_get_block_p(matrix=pao%matrix_H0, row=iatom, col=iatom, block=block_H0, found=found)
         CALL dbcsr_get_block_p(matrix=pao%matrix_N_diag, row=iatom, col=iatom, block=block_N, found=found)
         CPASSERT(ASSOCIATED(block_H0) .AND. ASSOCIATED(block_N))
         N = SIZE(block_U0, 1)

         ALLOCATE (block_V0(N, N))
         CALL pao_guess_initial_potential(qs_env, iatom, block_V0)

         ! construct H
         ALLOCATE (block_H(N, N))
         block_H = MATMUL(MATMUL(block_N, block_H0 + block_V0), block_N) ! transform into orthonormal basis

         ! diagonalize H
         ALLOCATE (H_evecs(N, N), H_evals(N))
         H_evecs = block_H
         CALL diamat_all(H_evecs, H_evals)

         ! use eigenvectors as initial guess
         block_U0 = H_evecs

         DEALLOCATE (block_H, H_evecs, H_evals, block_V0)
      END DO
      CALL dbcsr_iterator_stop(iter)
!$OMP END PARALLEL

      IF (pao%precondition) &
         CPABORT("PAO preconditioning not supported for selected parametrization.")

      CALL timestop(handle)
   END SUBROUTINE pao_param_init_exp

! **************************************************************************************************
!> \brief Finalize exponential parametrization
!> \param pao ...
! **************************************************************************************************
   SUBROUTINE pao_param_finalize_exp(pao)
      TYPE(pao_env_type), POINTER                        :: pao

      CALL dbcsr_release(pao%matrix_U0)

   END SUBROUTINE pao_param_finalize_exp

! **************************************************************************************************
!> \brief Returns the number of parameters for given atomic kind
!> \param qs_env ...
!> \param ikind ...
!> \param nparams ...
! **************************************************************************************************
   SUBROUTINE pao_param_count_exp(qs_env, ikind, nparams)
      TYPE(qs_environment_type), POINTER                 :: qs_env
      INTEGER, INTENT(IN)                                :: ikind
      INTEGER, INTENT(OUT)                               :: nparams

      INTEGER                                            :: cols, pao_basis_size, pri_basis_size, &
                                                            rows
      TYPE(gto_basis_set_type), POINTER                  :: basis_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      CALL get_qs_env(qs_env, qs_kind_set=qs_kind_set)
      CALL get_qs_kind(qs_kind_set(ikind), &
                       basis_set=basis_set, &
                       pao_basis_size=pao_basis_size)
      pri_basis_size = basis_set%nsgf

      ! we only consider rotations between occupied and virtuals
      rows = pao_basis_size
      cols = pri_basis_size - pao_basis_size
      nparams = rows*cols

   END SUBROUTINE pao_param_count_exp

! **************************************************************************************************
!> \brief Fills matrix_X with an initial guess
!> \param pao ...
! **************************************************************************************************
   SUBROUTINE pao_param_initguess_exp(pao)
      TYPE(pao_env_type), POINTER                        :: pao

      CALL dbcsr_set(pao%matrix_X, 0.0_dp) ! actual initial guess is matrix_U0

   END SUBROUTINE pao_param_initguess_exp

! **************************************************************************************************
!> \brief Takes current matrix_X and calculates the matrices A and B.
!> \param pao ...
!> \param qs_env ...
!> \param ls_scf_env ...
!> \param gradient ...
! **************************************************************************************************
   SUBROUTINE pao_calc_AB_exp(pao, qs_env, ls_scf_env, gradient)
      TYPE(pao_env_type), POINTER                        :: pao
      TYPE(qs_environment_type), POINTER                 :: qs_env
      TYPE(ls_scf_env_type), TARGET                      :: ls_scf_env
      LOGICAL, INTENT(IN)                                :: gradient

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pao_calc_AB_exp'

      INTEGER                                            :: handle
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s
      TYPE(dbcsr_type)                                   :: matrix_M, matrix_U

      CALL timeset(routineN, handle)
      CALL get_qs_env(qs_env, matrix_s=matrix_s)
      CALL dbcsr_create(matrix_U, matrix_type="N", dist=pao%diag_distribution, template=matrix_s(1)%matrix)
      CALL dbcsr_reserve_diag_blocks(matrix_U)

      !TODO: move this condition into pao_calc_U, use matrix_N as template
      IF (gradient) THEN
         CALL pao_calc_grad_lnv_wrt_U(qs_env, ls_scf_env, matrix_M)
         CALL pao_calc_U_exp(pao, matrix_U, matrix_M, pao%matrix_G)
         CALL dbcsr_release(matrix_M)
      ELSE
         CALL pao_calc_U_exp(pao, matrix_U)
      END IF

      CALL pao_calc_AB_from_U(pao, qs_env, ls_scf_env, matrix_U)
      CALL dbcsr_release(matrix_U)
      CALL timestop(handle)
   END SUBROUTINE pao_calc_AB_exp

! **************************************************************************************************
!> \brief Calculate new matrix U and optionally its gradient G
!> \param pao ...
!> \param matrix_U ...
!> \param matrix_M ...
!> \param matrix_G ...
! **************************************************************************************************
   SUBROUTINE pao_calc_U_exp(pao, matrix_U, matrix_M, matrix_G)
      TYPE(pao_env_type), POINTER                        :: pao
      TYPE(dbcsr_type)                                   :: matrix_U
      TYPE(dbcsr_type), OPTIONAL                         :: matrix_M, matrix_G

      CHARACTER(len=*), PARAMETER                        :: routineN = 'pao_calc_U_exp'

      COMPLEX(dp)                                        :: denom
      COMPLEX(dp), DIMENSION(:), POINTER                 :: evals
      COMPLEX(dp), DIMENSION(:, :), POINTER              :: block_D, evecs
      INTEGER                                            :: acol, arow, handle, i, iatom, j, k, M, &
                                                            N, nparams
      INTEGER, DIMENSION(:), POINTER                     :: blk_sizes_pao, blk_sizes_pri
      LOGICAL                                            :: found
      REAL(dp), DIMENSION(:, :), POINTER                 :: block_G, block_G_full, block_M, &
                                                            block_tmp, block_U, block_U0, block_X, &
                                                            block_X_full
      TYPE(dbcsr_iterator_type)                          :: iter

      CALL timeset(routineN, handle)

      CALL dbcsr_get_info(pao%matrix_Y, row_blk_size=blk_sizes_pri, col_blk_size=blk_sizes_pao)

!$OMP PARALLEL DEFAULT(NONE) SHARED(pao,matrix_U,matrix_M,matrix_G,blk_sizes_pri,blk_sizes_pao) &
!$OMP PRIVATE(iter,arow,acol,iatom,N,M,nparams,i,j,k,found) &
!$OMP PRIVATE(block_X,block_U,block_U0,block_X_full,evals,evecs) &
!$OMP PRIVATE(block_M,block_G,block_D,block_tmp,block_G_full,denom)
      CALL dbcsr_iterator_start(iter, pao%matrix_X)
      DO WHILE (dbcsr_iterator_blocks_left(iter))
         CALL dbcsr_iterator_next_block(iter, arow, acol, block_X)
         iatom = arow; CPASSERT(arow == acol)
         CALL dbcsr_get_block_p(matrix=matrix_U, row=iatom, col=iatom, block=block_U, found=found)
         CPASSERT(ASSOCIATED(block_U))
         CALL dbcsr_get_block_p(matrix=pao%matrix_U0, row=iatom, col=iatom, block=block_U0, found=found)
         CPASSERT(ASSOCIATED(block_U0))

         N = blk_sizes_pri(iatom) ! size of primary basis
         M = blk_sizes_pao(iatom) ! size of pao basis
         nparams = SIZE(block_X, 1)

         ! block_X stores only rotations between occupied and virtuals
         ! hence, we first have to build the full anti-symmetric exponent block
         ALLOCATE (block_X_full(N, N))
         block_X_full(:, :) = 0.0_dp
         DO i = 1, nparams
            block_X_full(MOD(i - 1, M) + 1, M + (i - 1)/M + 1) = +block_X(i, 1)
            block_X_full(M + (i - 1)/M + 1, MOD(i - 1, M) + 1) = -block_X(i, 1)
         END DO

         ! diagonalize block_X_full
         ALLOCATE (evals(N), evecs(N, N))
         CALL diag_antisym(block_X_full, evecs, evals)

         ! construct rotation matrix
         block_U(:, :) = 0.0_dp
         DO k = 1, N
            DO i = 1, N
               DO j = 1, N
                  block_U(i, j) = block_U(i, j) + REAL(EXP(evals(k))*evecs(i, k)*CONJG(evecs(j, k)), dp)
               END DO
            END DO
         END DO

         block_U = MATMUL(block_U0, block_U) ! prepend initial guess rotation

         ! TURNING POINT (if calc grad) ------------------------------------------
         IF (PRESENT(matrix_G)) THEN
            CPASSERT(PRESENT(matrix_M))

            CALL dbcsr_get_block_p(matrix=pao%matrix_G, row=iatom, col=iatom, block=block_G, found=found)
            CPASSERT(ASSOCIATED(block_G))
            CALL dbcsr_get_block_p(matrix=matrix_M, row=iatom, col=iatom, block=block_M, found=found)
            ! don't check ASSOCIATED(block_M), it might have been filtered out.

            ALLOCATE (block_D(N, N), block_tmp(N, N), block_G_full(N, N))
            DO i = 1, N
               DO j = 1, N
                  denom = evals(i) - evals(j)
                  IF (i == j) THEN
                     block_D(i, i) = EXP(evals(i)) ! diagonal elements
                  ELSE IF (ABS(denom) > 1e-10_dp) THEN
                     block_D(i, j) = (EXP(evals(i)) - EXP(evals(j)))/denom
                  ELSE
                     block_D(i, j) = 1.0_dp ! limit according to L'Hospital's rule
                  END IF
               END DO
            END DO

            IF (ASSOCIATED(block_M)) THEN
               block_tmp = MATMUL(TRANSPOSE(block_U0), block_M)
            ELSE
               block_tmp = 0.0_dp
            END IF
            block_G_full = fold_derivatives(block_tmp, block_D, evecs)

            ! return only gradient for rotations between occupied and virtuals
            DO i = 1, nparams
               block_G(i, 1) = 2.0_dp*block_G_full(MOD(i - 1, M) + 1, M + (i - 1)/M + 1)
            END DO

            DEALLOCATE (block_D, block_tmp, block_G_full)
         END IF

         DEALLOCATE (block_X_full, evals, evecs)

      END DO
      CALL dbcsr_iterator_stop(iter)
!$OMP END PARALLEL

      CALL timestop(handle)
   END SUBROUTINE pao_calc_U_exp

! **************************************************************************************************
!> \brief Helper routine, for calculating derivatives
!> \param M ...
!> \param D ...
!> \param R ...
!> \return ...
! **************************************************************************************************
   FUNCTION fold_derivatives(M, D, R) RESULT(G)
      REAL(dp), DIMENSION(:, :), INTENT(IN)              :: M
      COMPLEX(dp), DIMENSION(:, :), INTENT(IN)           :: D, R
      REAL(dp), DIMENSION(SIZE(M, 1), SIZE(M, 1))        :: G

      COMPLEX(dp), DIMENSION(:, :), POINTER              :: F, RF, RM, RMR
      INTEGER                                            :: n
      REAL(dp), DIMENSION(:, :), POINTER                 :: RFR

      n = SIZE(M, 1)

      ALLOCATE (RM(n, n), RMR(n, n), F(n, n), RF(n, n), RFR(n, n))

      RM = MATMUL(TRANSPOSE(CONJG(R)), TRANSPOSE(M))
      RMR = MATMUL(RM, R)
      F = RMR*D !Hadamard product
      RF = MATMUL(R, F)
      RFR = REAL(MATMUL(RF, TRANSPOSE(CONJG(R))), dp)

      ! gradient dE/dX has to be anti-symmetric
      G = 0.5_dp*(TRANSPOSE(RFR) - RFR)

      DEALLOCATE (RM, RMR, F, RF, RFR)
   END FUNCTION fold_derivatives

END MODULE pao_param_exp
