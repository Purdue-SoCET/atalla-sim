// discontinued

#ifndef __SIM_CLOCKED_OBJECT_HH__
#define __SIM_CLOCKED_OBJECT_HH__

#include "base/core.hh"
#include "base/clock_domain.hh"

namespace atalla
{ 

// Helper class that will be inherited by all objects that require a clock. Remember, SimObject and ClockedObject are orthogonal and will be derived by all simulated objects. 
class ClockedBase

    private: 
        
        mutable Tick tick; // next clock edge
        mutable Cycle cycles; // current cycle counter

        ClockDomain &clockDomain; 
        

    protected: 

        // register which clockDomain this should be a part of 
        ClockedBase(ClockDomain &clockDomain) { 
            clockDomain.register(this); 
        }

        // disable copying 
        Clocked(Clocked &) = delete;
        Clocked &operator=(Clocked &) = delete;

        // destructor
        virtual ~Clocked() { }
        
    public: 

        void updateClock() { 
            update(); 
            clockPeriodUpdated(); 
        }
        
}

#endif //__SIM_CLOCKED_OBJECT_HH__