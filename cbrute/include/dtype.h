#pragma once

namespace cbrute {

enum class PackDType { U8, U32, U64 };

inline int pack_width(PackDType pd) {
    switch (pd) {
        case PackDType::U8:  return 8;
        case PackDType::U32: return 32;
        case PackDType::U64: return 64;
    }
    return 8;
}

} // namespace cbrute
