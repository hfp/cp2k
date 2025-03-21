!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Tools common both to PME and SPME
!> \par History
!>      JGH (03-May-2001) : first correctly working version
!>      teo (Feb-2007)    : Merging common routines to spme and pme
!> \author JGH (21-Mar-2001)
! **************************************************************************************************
MODULE pme_tools

   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind
   USE cell_types,                      ONLY: cell_type,&
                                              real_to_scaled
   USE kinds,                           ONLY: dp
   USE particle_types,                  ONLY: particle_type
   USE realspace_grid_types,            ONLY: realspace_grid_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: get_center, set_list

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'pme_tools'

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param part ...
!> \param npart ...
!> \param center ...
!> \param p1 ...
!> \param rs ...
!> \param ipart ...
!> \param core_center ...
! **************************************************************************************************
   SUBROUTINE set_list(part, npart, center, p1, rs, ipart, core_center)

      TYPE(particle_type), DIMENSION(:), INTENT(IN)      :: part
      INTEGER, INTENT(IN)                                :: npart
      INTEGER, DIMENSION(:, :), INTENT(IN)               :: center
      INTEGER, INTENT(OUT)                               :: p1
      TYPE(realspace_grid_type), INTENT(IN)              :: rs
      INTEGER, INTENT(INOUT)                             :: ipart
      INTEGER, DIMENSION(:, :), OPTIONAL, POINTER        :: core_center

      INTEGER                                            :: ndim, npos
      INTEGER, DIMENSION(3)                              :: lb, ub
      REAL(KIND=dp)                                      :: charge
      TYPE(atomic_kind_type), POINTER                    :: atomic_kind

      p1 = 0
      lb = rs%lb_real
      ub = rs%ub_real

      DO
         ipart = ipart + 1
         IF (ipart > npart) EXIT
         atomic_kind => part(ipart)%atomic_kind
         CALL get_atomic_kind(atomic_kind=atomic_kind, qeff=charge)
         IF (charge == 0.0_dp .AND. part(ipart)%shell_index == 0) CYCLE
         IF (rs%desc%parallel) THEN
            ! check if the rs grid is distributed or not
            IF (ALL(rs%desc%group_dim == 1)) THEN
               ndim = rs%desc%group_size
               npos = rs%desc%my_pos
               ! All processors work on the same grid
               IF (MOD(ipart, ndim) == npos) THEN
                  p1 = ipart
                  EXIT
               END IF
            ELSE
               ! First check if this atom is on my grid
               IF (part(ipart)%shell_index /= 0 .AND. PRESENT(core_center)) THEN
                  IF (in_slice(core_center(:, part(ipart)%shell_index), lb, ub)) THEN
                     p1 = ipart
                  END IF
               ELSE
                  IF (in_slice(center(:, ipart), lb, ub)) THEN
                     p1 = ipart
                     EXIT
                  END IF
               END IF
            END IF
         ELSE
            p1 = ipart
            EXIT
         END IF
      END DO

   END SUBROUTINE set_list

! **************************************************************************************************
!> \brief ...
!> \param pos ...
!> \param lb ...
!> \param ub ...
!> \return ...
! **************************************************************************************************
   FUNCTION in_slice(pos, lb, ub) RESULT(internal)

      INTEGER, DIMENSION(3), INTENT(IN)                  :: pos, lb, ub
      LOGICAL                                            :: internal

      IF (ALL(pos >= lb) .AND. ALL(pos <= ub)) THEN
         internal = .TRUE.
      ELSE
         internal = .FALSE.
      END IF

   END FUNCTION in_slice

! **************************************************************************************************
!> \brief ...
!> \param part ...
!> \param box ...
!> \param center ...
!> \param delta ...
!> \param npts ...
!> \param n ...
! **************************************************************************************************
   SUBROUTINE get_center(part, box, center, delta, npts, n)

      TYPE(particle_type), DIMENSION(:), INTENT(IN)      :: part
      TYPE(cell_type), POINTER                           :: box
      INTEGER, DIMENSION(:, :), INTENT(OUT)              :: center
      REAL(KIND=dp), DIMENSION(:, :), INTENT(OUT)        :: delta
      INTEGER, DIMENSION(:), INTENT(IN)                  :: npts
      INTEGER, INTENT(IN)                                :: n

      INTEGER                                            :: ipart, mp
      REAL(KIND=dp)                                      :: rmp
      REAL(KIND=dp), DIMENSION(3)                        :: ca, gp, s

      ! The pbc algorithm is sensitive to numeric noise and compiler optimization because of ANINT.
      ! Therefore center and delta have to be computed simultaneously to ensure they are consistent.
      mp = MAXVAL(npts(:))
      rmp = REAL(mp, KIND=dp)
      DO ipart = 1, SIZE(part)
         ! compute the scaled coordinate of atom ipart
         CALL real_to_scaled(s, part(ipart)%r, box)
         s = s - ANINT(s)
         ! find the continuous ``grid'' point
         gp = REAL(npts, KIND=dp)*s
         ! find the closest grid point (on big grid)
         IF (MOD(n, 2) == 0) THEN
            center(:, ipart) = INT(gp + rmp) - mp
            ca(:) = REAL(center(:, ipart), KIND=dp) + 0.5_dp
         ELSE
            center(:, ipart) = NINT(gp)
            ca(:) = REAL(center(:, ipart), KIND=dp)
         END IF
         center(:, ipart) = center(:, ipart) + npts(:)/2
         center(:, ipart) = MODULO(center(:, ipart), npts(:))
         center(:, ipart) = center(:, ipart) - npts(:)/2
         ! find the distance vector
         delta(:, ipart) = gp - ca(:)
      END DO

   END SUBROUTINE get_center

END MODULE pme_tools

