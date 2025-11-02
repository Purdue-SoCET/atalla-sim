// discontinued

#ifndef __SIM_TYPES_HH__
#define __SIM_TYPES_HH__


namespace atalla 
{ 

    // Counters, for whatever usecase you have.
    typedef uint64_t LargeCounter; 
    typedef uint32_t ShortCounter; 

    // Tick Type
    typedef uint64_t Tick;

    // Largest Address Size
    typedef uint32_t Addr;

    class Cycles 
    { 
    
        private: 
            uint64_t count; 

        public: 
            // default 
            Cycles() : count(0) { }

            // Set a value
            explicit constexpr Cycles(uint64_t count_) : count(count_) { } 

            // Force static_cast<>(object); 
            explicit constexpr operator uint64_t() const { return count; }

            Cycles& operator++() { ++count; return *this; }
            Cycles& operator--() { assert(count != 0); --count; return *this; }

    }
}

#endif //__SIM_TYPES_HH__
