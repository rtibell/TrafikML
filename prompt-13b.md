# Task
- Please change the MLP task to instead of calculating the speed it should predict what the statusEnum is from one of the four possibilities: "freeflow", "heavy", "congested", "impossible"
- Use a soft max to determine statusEnum value. 
- Code the statusEnum as:
	freeflow" = 0
	"heavy" = 1
	"congested" = 2
	"impossible" = 3
- If possible add support for Apple GPU on MacBook M2