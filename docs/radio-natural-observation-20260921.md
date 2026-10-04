# Natural held-arm observation and axial retry

behavior-radio-button 0.1.22 keeps the public input model and all Ability Action contracts. SDK, Ability and Runtime are unchanged. Old chest fixed-point parameters remain accepted; the natural-posture flow uses limb geometry.

Flow: torso joint3 upright (existing 3 s trajectory), natural upper-arm-down / forearm-forward posture, close operating gripper, optical look-at observation, button localization and existing button-position approach. Only OBJECT_NOT_FOUND triggers one held-arm axial 180-degree roll and a second observation. Plans and verified actions are checkpointed. The operator stays at its observation pose during the roll, then replans from measured holding-hand pose.

Motion is segmented through existing MoveArmJoint. Terminal command_timeout may be accepted by Skill only after measured joints are within 0.02 rad, successive readings differ by at most 0.001 rad, and measured EEF is within 3 cm / 4 degrees of FK target. At most three samples, 0.25 s apart. Original failed Action remains in evidence. Other errors end execution.

Observation center is approximated as holding EEF local [0,0,0.13] m. This is a radio-grasp geometry assumption. Camera extrinsics are from native left/right calibration. Constraints verify optical look-at, joint limits and workspace region; full swept collision planning remains outside this change.

Evidence before this change: generation 13 natural left pose physically reached; right wrist then detected red button with confidence 0.875. Fixed-position left axial roll physically reached 179.30 degrees, final position drift 2.20 mm, maximum sampled drift 3.53 mm; object-to-hand relative translation changed 0.038 mm. Those motions used Runtime continuous sequences. The updated Skill uses segmented existing Ability actions and measured timeout recovery; full physical Skill execution remains for the next user trial.

54 local tests passed: both-side natural pose / optical look-at / axial IK path, Mock flow and replay, bounded timeout recovery acceptance/rejection, package boundaries and existing evidence handling. No version-number or literal configuration-value tests added.

## 0.1.23: held-arm timeout recovery tolerance

The latest execution rex-ab55564a-15c2-4b5c-90a8-7cf6ac16def3 stopped at natural-posture segment 13/21. After terminal timeout, recorded joint5 error stabilized at 0.02080 rad, successive joint difference 0.000105 rad, EEF error 4.26 mm / 1.72 degrees. User authorized changing the Skill recovery joint tolerance to 0.03 rad. Stability, EEF, torso and underlying controller thresholds remain unchanged. The recorded checkpoint replay now passes; 11 action-flow tests pass. No new physical actions were sent during this update.

## 0.1.24: independently plan button approach orientation

Execution rex-27bee626-d42d-48b0-acd3-c2eeec7b4626 localized the button on the first view (confidence 0.8552), then timed out approaching it. Position error stayed near 36 mm from simulation second 2 through second 30; right joint2 reached its upper limit. Ideal native weighted-IK replay with the old observation orientation reproduced a 36.73 mm residual.

The Skill now reads fresh operator EEF, torso and joints after recognition, fixes the recognized button position, and optimizes wrist orientation. Feasible candidates must have 5-degree joint-limit margin and point EEF +Z within 60 degrees of the observation-to-button direction. The objective prefers a small orientation change and nearby joint posture. The resulting pose retains the button observation revision and uses the existing MoveEndEffector action. Pose and diagnostics are checkpointed. The surface normal, contact and switch state are not inferred by these constraints.

On the recorded target, the selected orientation changes by 25.04 degrees. Ideal native IK replay from the observation posture reaches 0.098 mm / 0.00025 degrees error. The native solver chooses its own joint branch, with 1.63-degree final margin in this replay; the planner candidate's 5-degree margin does not guarantee native trajectory margin. Collision, load and actual pressing remain for physical validation.

22 tests passed: recorded failed goal, both arm models, unreachable target, first/second detection, error propagation, final-pose checkpoint reuse, and package boundaries. Existing public inputs and all SDK/Ability interfaces remain unchanged. No physical actions were issued during this update.
