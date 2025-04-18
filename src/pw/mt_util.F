!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
MODULE mt_util
   USE bibliography,                    ONLY: Martyna1999,&
                                              cite_reference
   USE kinds,                           ONLY: dp
   USE mathconstants,                   ONLY: fourpi,&
                                              oorootpi,&
                                              pi
   USE pw_grid_types,                   ONLY: pw_grid_type
   USE pw_methods,                      ONLY: pw_axpy,&
                                              pw_transfer,&
                                              pw_zero
   USE pw_pool_types,                   ONLY: pw_pool_create,&
                                              pw_pool_release,&
                                              pw_pool_type
   USE pw_types,                        ONLY: pw_c1d_gs_type,&
                                              pw_r3d_rs_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE
   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'mt_util'

   INTEGER, PARAMETER, PUBLIC               :: MT2D = 1101, &
                                               MT1D = 1102, &
                                               MT0D = 1103

   PUBLIC :: MTin_create_screen_fn
CONTAINS

! **************************************************************************************************
!> \brief Initialize the Martyna && Tuckerman Poisson Solver
!> \param screen_function ...
!> \param pw_pool ...
!> \param method ...
!> \param alpha ...
!> \param special_dimension ...
!> \param slab_size ...
!> \param super_ref_pw_grid ...
!> \author Teodoro Laino (16.06.2004)
! **************************************************************************************************
   SUBROUTINE MTin_create_screen_fn(screen_function, pw_pool, method, alpha, &
                                    special_dimension, slab_size, super_ref_pw_grid)
      TYPE(pw_c1d_gs_type), POINTER                      :: screen_function
      TYPE(pw_pool_type), POINTER                        :: pw_pool
      INTEGER, INTENT(IN)                                :: method
      REAL(KIND=dp), INTENT(in)                          :: alpha
      INTEGER, INTENT(IN)                                :: special_dimension
      REAL(KIND=dp), INTENT(in)                          :: slab_size
      TYPE(pw_grid_type), POINTER                        :: super_ref_pw_grid

      CHARACTER(len=*), PARAMETER :: routineN = 'MTin_create_screen_fn'

      INTEGER                                            :: handle, ig, iz
      REAL(KIND=dp)                                      :: alpha2, g2, g3d, gxy, gz, zlength
      TYPE(pw_c1d_gs_type), POINTER                      :: Vlocg
      TYPE(pw_pool_type), POINTER                        :: pw_pool_aux
      TYPE(pw_r3d_rs_type), POINTER                      :: Vloc

      CALL timeset(routineN, handle)
      NULLIFY (Vloc, Vlocg, pw_pool_aux)
      !
      ! For Martyna-Tuckerman we set up an auxiliary pw_pool at an higher cutoff
      !
      CALL cite_reference(Martyna1999)
      IF (ASSOCIATED(super_ref_pw_grid)) THEN
         CALL pw_pool_create(pw_pool_aux, pw_grid=super_ref_pw_grid)
      END IF
      NULLIFY (screen_function)
      ALLOCATE (screen_function)
      CALL pw_pool%create_pw(screen_function)
      CALL pw_zero(screen_function)
      SELECT CASE (method)
      CASE (MT0D)
         NULLIFY (Vloc, Vlocg)
         ALLOCATE (Vloc, Vlocg)
         IF (ASSOCIATED(pw_pool_aux)) THEN
            CALL pw_pool_aux%create_pw(Vloc)
            CALL pw_pool_aux%create_pw(Vlocg)
         ELSE
            CALL pw_pool%create_pw(Vloc)
            CALL pw_pool%create_pw(Vlocg)
         END IF
         CALL mt0din(Vloc, alpha)
         CALL pw_transfer(Vloc, Vlocg)
         CALL pw_axpy(Vlocg, screen_function)
         IF (ASSOCIATED(pw_pool_aux)) THEN
            CALL pw_pool_aux%give_back_pw(Vloc)
            CALL pw_pool_aux%give_back_pw(Vlocg)
         ELSE
            CALL pw_pool%give_back_pw(Vloc)
            CALL pw_pool%give_back_pw(Vlocg)
         END IF
         DEALLOCATE (Vloc, Vlocg)
         !
         ! Get rid of the analytical FT of the erf(a*r)/r
         !
         alpha2 = alpha*alpha
         DO ig = screen_function%pw_grid%first_gne0, screen_function%pw_grid%ngpts_cut_local
            g2 = screen_function%pw_grid%gsq(ig)
            g3d = fourpi/g2
            screen_function%array(ig) = screen_function%array(ig) - g3d*EXP(-g2/(4.0E0_dp*alpha2))
         END DO
         IF (screen_function%pw_grid%have_g0) &
            screen_function%array(1) = screen_function%array(1) + fourpi/(4.0E0_dp*alpha2)
      CASE (MT2D)
         iz = special_dimension ! iz is the direction with NO PBC
         zlength = slab_size ! zlength is the thickness of the cell
         DO ig = screen_function%pw_grid%first_gne0, screen_function%pw_grid%ngpts_cut_local
            gz = screen_function%pw_grid%g(iz, ig)
            g2 = screen_function%pw_grid%gsq(ig)
            gxy = SQRT(ABS(g2 - gz*gz))
            g3d = fourpi/g2
            screen_function%array(ig) = -g3d*COS(gz*zlength/2.0_dp)*EXP(-gxy*zlength/2.0_dp)
         END DO
         IF (screen_function%pw_grid%have_g0) screen_function%array(1) = pi*zlength*zlength/2.0_dp
      CASE (MT1D)
         iz = special_dimension ! iz is the direction with PBC
         CALL mt1din(screen_function)
         CPABORT("MT1D unimplemented")
      END SELECT
      CALL pw_pool_release(pw_pool_aux)
      CALL timestop(handle)
   END SUBROUTINE MTin_create_screen_fn

! **************************************************************************************************
!> \brief Calculates the Tuckerman Green's function in reciprocal space
!>      according the scheme published on:
!>      Martyna and Tuckerman, J. Chem. Phys. Vol. 110, No. 6, 2810-2821
!> \param Vloc ...
!> \param alpha ...
!> \author Teodoro Laino (09.03.2005)
! **************************************************************************************************
   SUBROUTINE mt0din(Vloc, alpha)
      TYPE(pw_r3d_rs_type), POINTER                      :: Vloc
      REAL(KIND=dp), INTENT(in)                          :: alpha

      CHARACTER(len=*), PARAMETER                        :: routineN = 'mt0din'

      INTEGER                                            :: handle, i, ii, j, jj, k, kk
      INTEGER, DIMENSION(:), POINTER                     :: glb
      INTEGER, DIMENSION(:, :), POINTER                  :: bo
      REAL(KIND=dp)                                      :: dx, dy, dz, fact, omega, r, r2, x, y, &
                                                            y2, z, z2
      REAL(KIND=dp), DIMENSION(3)                        :: box, box2
      TYPE(pw_grid_type), POINTER                        :: grid

      CALL timeset(routineN, handle)

      grid => Vloc%pw_grid
      bo => grid%bounds_local
      glb => grid%bounds(1, :)
      Vloc%array = 0.0_dp
      box = REAL(grid%npts, kind=dp)*grid%dr
      box2 = box/2.0_dp
      omega = PRODUCT(box)
      fact = omega
      dx = grid%dr(1)
      dy = grid%dr(2)
      dz = grid%dr(3)
      kk = bo(1, 3)
      DO k = bo(1, 3), bo(2, 3)
         z = REAL(k - glb(3), dp)*dz; IF (z .GT. box2(3)) z = box(3) - z
         z2 = z*z
         jj = bo(1, 2)
         DO j = bo(1, 2), bo(2, 2)
            y = REAL(j - glb(2), dp)*dy; IF (y .GT. box2(2)) y = box(2) - y
            y2 = y*y
            ii = bo(1, 1)
            DO i = bo(1, 1), bo(2, 1)
               x = REAL(i - glb(1), dp)*dx; IF (x .GT. box2(1)) x = box(1) - x
               r2 = x*x + y2 + z2
               r = SQRT(r2)
               IF (r .GT. 1.0E-10_dp) THEN
                  Vloc%array(ii, jj, kk) = erf(alpha*r)/r*fact
               ELSE
                  Vloc%array(ii, jj, kk) = 2.0_dp*alpha*oorootpi*fact
               END IF
               ii = ii + 1
            END DO
            jj = jj + 1
         END DO
         kk = kk + 1
      END DO
      CALL timestop(handle)
   END SUBROUTINE Mt0din

! **************************************************************************************************
!> \brief Calculates the Tuckerman Green's function in reciprocal space
!>      according the scheme published on:
!>      Martyna and Tuckerman, J. Chem. Phys. Vol. 121, No. 23, 11949
!> \param screen_function ...
!> \author Teodoro Laino (11.2005)
! **************************************************************************************************
   SUBROUTINE mt1din(screen_function)
      TYPE(pw_c1d_gs_type), POINTER                      :: screen_function

      CHARACTER(len=*), PARAMETER                        :: routineN = 'mt1din'

      INTEGER                                            :: handle
      REAL(KIND=dp)                                      :: dx, dy, dz, omega
      REAL(KIND=dp), DIMENSION(3)                        :: box, box2
      TYPE(pw_grid_type), POINTER                        :: grid

      CALL timeset(routineN, handle)
      grid => screen_function%pw_grid
      box = REAL(grid%npts, kind=dp)*grid%dr
      box2 = box/2.0_dp
      omega = PRODUCT(box)
      dx = grid%dr(1)
      dy = grid%dr(2)
      dz = grid%dr(3)

      CALL timestop(handle)
   END SUBROUTINE mt1din

END MODULE mt_util
