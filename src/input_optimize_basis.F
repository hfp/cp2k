!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief builds the input structure for optimize_basis
!> \par History
!>      03.2012 created [Florian Schiffmann]
!> \author Florian Schiffmann
! **************************************************************************************************
MODULE input_optimize_basis

   USE input_constants,                 ONLY: do_opt_all,&
                                              do_opt_coeff,&
                                              do_opt_exps,&
                                              do_opt_none
   USE input_keyword_types,             ONLY: keyword_create,&
                                              keyword_release,&
                                              keyword_type
   USE input_section_types,             ONLY: section_add_keyword,&
                                              section_add_subsection,&
                                              section_create,&
                                              section_release,&
                                              section_type
   USE input_val_types,                 ONLY: char_t,&
                                              integer_t,&
                                              real_t
   USE kinds,                           ONLY: dp
   USE string_utilities,                ONLY: s2a
#include "./base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'input_optimize_basis'
   PUBLIC :: create_optimize_basis_section

CONTAINS

! **************************************************************************************************
!> \brief creates the optimize_basis section
!> \param section ...
!> \author Florian Schiffmann
! **************************************************************************************************
   SUBROUTINE create_optimize_basis_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: subsection

      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="OPTIMIZE_BASIS", &
                          description="describes a basis optimization job, in which an ADMM like approach is used to"// &
                          " find the best exponents and/or coefficients to match a given training set.", &
                          repeats=.FALSE.)
      NULLIFY (keyword, subsection)

      CALL keyword_create(keyword, __LOCATION__, name="BASIS_TEMPLATE_FILE", &
                          description="Name of the basis set file, containing the structure of the new basis set", &
                          usage="BASIS_TEMPLATE_FILE <FILENAME>", &
                          type_of_var=char_t, repeats=.FALSE., &
                          default_c_val="BASIS_SET", n_var=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BASIS_WORK_FILE", &
                          description="Name of the basis set file which is created to be read as initial guess", &
                          usage="BASIS_WORK_FILE <FILENAME>", &
                          type_of_var=char_t, repeats=.FALSE., &
                          default_c_val="BASIS_WORK_FILE", n_var=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BASIS_OUTPUT_FILE", &
                          description="Name of the basis set file containing the optimized basis", &
                          usage="BASIS_OUTPUT_FILE <FILENAME>", &
                          type_of_var=char_t, repeats=.FALSE., &
                          default_c_val="BASIS_OUTPUT_FILE", n_var=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="WRITE_FREQUENCY", &
                          description="Frequency at which the intermediate results should be written", &
                          usage="WRITE_FREQUENCY 1000", &
                          default_i_val=5000)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="USE_CONDITION_NUMBER", &
                          description="Determines whether condition number should be part of optimization or not", &
                          usage="USE_CONDITION_NUMBER", &
                          default_l_val=.FALSE., lone_keyword_l_val=.TRUE.)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create( &
         keyword, __LOCATION__, name="BASIS_COMBINATIONS", &
         description="If multiple atomic kinds are fitted at the same time, this keyword "// &
         "allows to specify which basis sets should be used together in optimization (underived set ID=0). "// &
         "If skipped all combinations are used. The order is taken as the kinds and sets are specified in the input", &
         repeats=.TRUE., &
         usage="BASIS_COMBINATIONS SET_ID(KIND1) SET_ID(KIND2) ... ", type_of_var=integer_t, n_var=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create( &
         keyword, __LOCATION__, name="RESIDUUM_WEIGHT", &
         description="This keyword allows to give different weight factors to the "// &
         "residuum of the different basis combinations. "// &
         "The first entry corresponds to the original basis sets. Every further value is assigned to the combinations "// &
         "in the order given for BASIS_COMBINATIONS.", &
         repeats=.TRUE., &
         usage="RESIDUUM_WEIGHT REAL ", default_r_val=1.0_dp)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create( &
         keyword, __LOCATION__, name="CONDITION_WEIGHT", &
         description="This keyword allows to give different weight factors to the "// &
         "condition number of different basis combinations (LOG(cond) is used). "// &
         "The first entry corresponds to the original basis sets. Every further value is assigned to the combinations "// &
         "in the order given for BASIS_COMBINATIONS.", &
         repeats=.TRUE., &
         usage="CONDITION_WEIGHT REAL ", default_r_val=1.0_dp)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="GROUP_PARTITION", &
                          description="Allows the specification of the group mpi group sizes in parallel "// &
                          "runs. If less Groups than tasks are speciefied, consecutive calculations "// &
                          "Will be assigned to one group (derived basis sets and then training sets) "// &
                          "If keyword is skipped, equal group sizes will be generated trying to fit all calculations.", &
                          repeats=.TRUE., &
                          usage="GROUP_PARTITION INT INT ... ", type_of_var=integer_t, n_var=-1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL create_fit_kinds_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

      CALL create_training_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

      CALL create_powell_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

   END SUBROUTINE create_optimize_basis_section

! **************************************************************************************************
!> \brief ...
!> \param section ...
! **************************************************************************************************
   SUBROUTINE create_fit_kinds_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword
      TYPE(section_type), POINTER                        :: subsection

      NULLIFY (keyword, subsection)
      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="FIT_KIND", &
                          description="specicifies the atomic kinds to be fitted and the basis"// &
                          " sets associated with the kind.", &
                          repeats=.TRUE.)

      CALL keyword_create(keyword, __LOCATION__, name="_SECTION_PARAMETERS_", &
                          description="The name of the kind described in this section.", &
                          usage="H", default_c_val="DEFAULT")
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BASIS_SET", &
                          description="The name of the basis set for the kind. Has to be specified in BASIS_TEMPLATE_FILE.", &
                          usage="BASIS_SET H", default_c_val="DEFAULT")
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="INITIAL_DEGREES_OF_FREEDOM", &
                          description="Specifies the initial degrees of freedom in the basis optimization. "// &
                          "This can be used to make further specifications easier", &
                          usage="INITIAL_DEGREES_OF_FREEDOM ALL", &
                          enum_c_vals=s2a("ALL", "NONE", "COEFFICIENTS", "EXPONENTS"), &
                          enum_desc=s2a("Set all parameters in the basis to be variable.", &
                                        "Set all parameters in the basis to be fixed.", &
                                        "Set all coefficients in the basis set to be variable.", &
                                        "Set all exponents in the basis to be variable."), &
                          enum_i_vals=(/do_opt_all, do_opt_none, do_opt_coeff, do_opt_exps/), &
                          default_i_val=do_opt_coeff)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="SWITCH_COEFF_STATE", &
                          description="Allows to switch the state of a given coefficient from current state "// &
                          "(varibale/fixed)) to the opposite state. The three integers indicate "// &
                          "the set number, the angular momentum i'th contraction and i'th coefficient", repeats=.TRUE., &
                          usage="SWITCH_COEFF_STATE SET L CONTRACTION IPGF", type_of_var=integer_t, n_var=4)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="SWITCH_CONTRACTION_STATE", &
                          description="Allows to switch the state of a given contraction from current state "// &
                          "(varibale/fixed)) to the opposite state. The three integers indicate "// &
                          "the set number, the angular momentum and i'th contraction ", repeats=.TRUE., &
                          usage="SWITCH_CONTRACTION_STATE SET L CONTRACTION ", type_of_var=integer_t, n_var=3)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="SWITCH_EXP_STATE", &
                          description="Allows to switch the state of a given exponent from current state "// &
                          "(varibale/fixed)) to the opposite state. The two integers indicate "// &
                          "the set number and i'th exponent", repeats=.TRUE., &
                          usage="SWITCH_EXP_STATE SET IEXP", type_of_var=integer_t, n_var=2)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="SWITCH_SET_STATE", &
                          description="Allows to switch the states of in a set from current state "// &
                          "(varibale/fixed)) to the opposite state. The two integers indicate "// &
                          "the affected part (0=ALL,1=EXPS,2=COEFF) and i'th set", repeats=.TRUE., &
                          usage="SWITCH_SET_STATE SET IEXP", type_of_var=integer_t, n_var=2)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL create_constrain_exp_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

      CALL create_derived_sets_section(subsection)
      CALL section_add_subsection(section, subsection)
      CALL section_release(subsection)

   END SUBROUTINE create_fit_kinds_section

! **************************************************************************************************
!> \brief ...
!> \param section ...
! **************************************************************************************************
   SUBROUTINE create_derived_sets_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword

      NULLIFY (keyword)
      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="DERIVED_BASIS_SETS", &
                          description="This section can be used to create subsets of a basis"// &
                          " which will be fitted at the same time. This is especially useful if connected"// &
                          " bsis sets e.g. TZVP, DZVP, SZV should be fitted.", &
                          repeats=.TRUE.)

      CALL keyword_create(keyword, __LOCATION__, name="BASIS_SET_NAME", &
                          description="Defines the name of the derived basis set, which will be "// &
                          "automatically generated otherwise.", &
                          usage="BASIS_SET_NAME {word}", &
                          type_of_var=char_t, &
                          repeats=.FALSE., &
                          default_c_val="")
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="REFERENCE_SET", &
                          description="Specifies the reference basis ID which is used as template to create the new set. "// &
                          "The original basis has ID 0. All following sets are counted in order as specified in the Input."// &
                          " The descriptors always assume the structure of the input basis set.", &
                          repeats=.FALSE., usage="REFERENCE_SET INTEGER", default_i_val=0)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="REMOVE_CONTRACTION", &
                          description="Can be used to remove a contraction from the reference basis set. "// &
                          "The contraction is speciefied by set number, angular momentum and number of contraction."// &
                          " The descriptors always assume the structure of the input basis set.", &
                          repeats=.TRUE., usage="REMOVE_CONTRACTION SET L ICONTRACTION", type_of_var=integer_t, n_var=3)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="REMOVE_SET", &
                          description="Can be used to remove a set from the reference basis set. ", &
                          repeats=.TRUE., usage="REMOVE_SET SET", type_of_var=integer_t, n_var=1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

   END SUBROUTINE create_derived_sets_section

! **************************************************************************************************
!> \brief ...
!> \param section ...
! **************************************************************************************************
   SUBROUTINE create_constrain_exp_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword

      NULLIFY (keyword)
      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="CONSTRAIN_EXPONENTS", &
                          description="specicifies constraints for the exponents to be fitted."// &
                          " Only a single constraint can be applied to an exponent", &
                          repeats=.TRUE.)

      CALL keyword_create(keyword, __LOCATION__, name="USE_EXP", &
                          description="Defines the exponent to be constraint. The two integers indicate "// &
                          "the set number and i'th exponent. The value -1 can be used to mark all sets/exponents in a set.", &
                          repeats=.FALSE., usage="USE_EXP SET IEXP", type_of_var=integer_t, n_var=2)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="BOUNDARIES", &
                          description="Defines the boundaries to which the optimization is restricted."// &
                          " First value is the lower bound, second value is the upper bound.", &
                          repeats=.FALSE., usage="BOUNDARIES LOWER UPPER", type_of_var=real_t, n_var=2)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="MAX_VAR_FRACTION", &
                          description="Defines the maximum fractionr by which the exponent is allowed to vary."// &
                          " e.g. 0.5 allows the exp to vary by 0.5*exp in both directions.", &
                          repeats=.FALSE., usage="MAX_VAR_FRACTION REAL", type_of_var=real_t, n_var=1)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

   END SUBROUTINE create_constrain_exp_section

! **************************************************************************************************
!> \brief ...
!> \param section ...
! **************************************************************************************************
   SUBROUTINE create_training_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword

      NULLIFY (keyword)
      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="TRAINING_FILES", &
                          description="specicifies the location in which the files necessary for"// &
                          " fitting procedure are located. Each Training set needs a repetition of this section.", &
                          repeats=.TRUE.)

      CALL keyword_create(keyword, __LOCATION__, name="DIRECTORY", &
                          description="the directory in which the files are placed", &
                          usage="DIRECTORY /my/path", &
                          default_lc_val=".")
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="INPUT_FILE_NAME", &
                          description="the filename of the input file used to run the original calculation", &
                          usage="INPUT_FILE_NAME my_input.inp", &
                          default_lc_val="input.inp")
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

   END SUBROUTINE create_training_section

! **************************************************************************************************
!> \brief ...
!> \param section ...
! **************************************************************************************************
   SUBROUTINE create_powell_section(section)
      TYPE(section_type), POINTER                        :: section

      TYPE(keyword_type), POINTER                        :: keyword

      NULLIFY (keyword)
      CPASSERT(.NOT. ASSOCIATED(section))
      CALL section_create(section, __LOCATION__, name="OPTIMIZATION", &
                          description="sets the parameters for optimizition, output frequency and restarts", &
                          repeats=.FALSE.)

      CALL keyword_create(keyword, __LOCATION__, name="ACCURACY", &
                          description="Final accuracy requested in optimization (RHOEND)", &
                          usage="ACCURACY 0.00001", &
                          default_r_val=1.e-5_dp)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="STEP_SIZE", &
                          description="Initial step size for search algorithm (RHOBEG)", &
                          usage="STEP_SIZE 0.005", &
                          default_r_val=0.1_dp)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

      CALL keyword_create(keyword, __LOCATION__, name="MAX_FUN", &
                          description="Maximum number of function evaluations", &
                          usage="MAX_FUN 1000", &
                          default_i_val=5000)
      CALL section_add_keyword(section, keyword)
      CALL keyword_release(keyword)

   END SUBROUTINE create_powell_section

END MODULE input_optimize_basis
