### Overview 

The core ideas of the Atalla-Sim are based off Gem5. Let's go through them one-by-one: 

- Tick
    Global counter. Monotonically increases. 
- Clock 
    Local counter. Monotonically increases. 
- ClockPeriod
    Ticks per cycle. 
    If ClockPeriod(500), think 500 ticks. Clock edges are 500, 1000, etc.
    Frequency = Tick/ClockPeriod
- ClockedBase 
    Stores a tick - tick timestamp of the next clock edge for the object. 
    Stores a cycle - cycle index aligning with tick. Cycle whose edge is at the tick. 
    tick = cycle * ClockPeriod
- ClockedObject 
    