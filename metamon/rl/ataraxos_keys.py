"""Reserved replay fields for the Epoch475 Ataraxos experiment."""

SOURCE = "_ataraxos_online_source"
BEHAVIOR_LOG_PROBS = "_ataraxos_behavior_log_probs"
BEHAVIOR_VALUE = "_ataraxos_behavior_value"
BEHAVIOR_VALID = "_ataraxos_behavior_valid"
BEHAVIOR_LEGAL = "_ataraxos_behavior_legal"
ADVANTAGE = "_ataraxos_advantage"
VALUE_TARGET = "_ataraxos_value_target"
OUTCOME = "_ataraxos_outcome"
ADV_THRESHOLD = "_ataraxos_adv_threshold"
OFFLINE_ACTOR_WEIGHT = "_ataraxos_offline_actor_weight"
# Optimizer update count of the behavior policy that collected an online
# replay (-1 for offline data or replays that predate the field). Used only
# for data-age diagnostics.
BEHAVIOR_UPDATE = "_ataraxos_behavior_update"
