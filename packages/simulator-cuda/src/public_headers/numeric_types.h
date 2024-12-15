#ifndef NUMERIC_TYPES_H
#define NUMERIC_TYPES_H

// CURRENCY
#define CURRENCY double
#define CURRENCY_MAX_VALUE 1.7976931348623157E+308

// -----------------------------------------------------------------------------
// EFFICIENT MODE
// -----------------------------------------------------------------------------
// FLOAT
#define FLOAT float
#define FLOAT_L(x) (x##f)
#define FLOAT_MA fmaf
#define FLOAT_MAX fmaxf
#define FLOAT_MIN fminf
#define FLOAT_POWF powf
#define __FLOAT_POWF __powf
#define __FLOAT_SATURATE __saturatef
#define FLOAT_MAX_VALUE 3.402823466e+38f
#define FLOAT_LOG1P log1pf
#define __FLOAT_LOG __logf
#define FLOAT_EXPM1 expm1f
#define __FLOAT_EXP __expf
#define FLOAT2 float2
#define FLOAT4 float4
#define FLOAT_DIVIDE fdividef
#define __FLOAT_DIVIDE __fdividef

// CURRENCY_NPV
#define CURRENCY_NPV float
#define CURRENCY_NPV_L(x) (x##f)
#define CURRENCY_NPV_MA fma

// -----------------------------------------------------------------------------
// REPLICATION MODE
// -----------------------------------------------------------------------------
// FLOAT
// #define FLOAT double
// #define FLOAT_L(x) (x)
// #define FLOAT_MA fma
// #define FLOAT_MAX fmax
// #define FLOAT_MIN fmin
// #define FLOAT_POWF pow
// #define __FLOAT_DIVIDE FLOAT_DIVIDE
// #define __FLOAT_SATURATE(x) max(0.0, min(x, 1.0))
// #define FLOAT_MAX_VALUE 1.7976931348623157E+308
// #define FLOAT_LOG1P log1p
// #define __FLOAT_LOG log
// #define FLOAT_EXPM1 expm1
// #define __FLOAT_EXP exp
// #define FLOAT2 double2
// #define FLOAT4 double4
// #define __FLOAT_POWF FLOAT_POWF
// #define FLOAT_DIVIDE(x, y) ((x) / (y))

// // CURRENCY_NPV
// #define CURRENCY_NPV double
// #define CURRENCY_NPV_L(x) (x)
// #define CURRENCY_NPV_MA fma

#endif // NUMERIC_TYPES_H
