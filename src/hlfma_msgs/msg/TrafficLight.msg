# SPEC §1.1: trafficLights 는 1칸 고정. 없으면 (0, 0) 이 온다.
int32 id

uint8 STATE_UNASSIGNED = 0
uint8 STATE_RED        = 1
uint8 STATE_YELLOW     = 2
uint8 STATE_GREEN      = 3
uint8 STATE_LEFT       = 4
uint8 STATE_GREEN_LEFT = 5
uint8 STATE_FLASH      = 6
uint8 state
