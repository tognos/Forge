#import <UIKit/UIKit.h>

//! Project version number for Forge.
FOUNDATION_EXPORT double ForgeVersionNumber;

//! Project version string for Forge.
FOUNDATION_EXPORT const unsigned char ForgeVersionString[];

// In this header, you should import all the public headers of your framework using statements like #import <Forge/PublicHeader.h>

static inline void storeAsF16(float value, uint16_t *pointer) { *(__fp16 *)pointer = value; }
static inline float loadFromF16(const uint16_t *pointer) { return *(const __fp16 *)pointer; }
