export declare type JSONTest<To extends From, From = unknown> = (x: From) => x is To;
export declare type JSONGuardResult<To> = {
    error: false;
    value: To;
} | {
    error: true;
    message: string;
};
export declare type JSONGuard<To extends From, From = unknown> = (x: From) => JSONGuardResult<To>;
export declare type JSONGuardConstructor<Options, To extends From, From = unknown> = (options: Options) => JSONGuard<To, From>;
export declare function chain<T0, T1 extends T0, T2 extends T1>(...tests: [JSONGuard<T1, T0>, JSONGuard<T2, T1>]): JSONGuard<T2, T0>;
export declare function chain<T0, T1 extends T0, T2 extends T1, T3 extends T2>(...tests: [JSONGuard<T1, T0>, JSONGuard<T2, T1>, JSONGuard<T3, T2>]): JSONGuard<T3, T0>;
export declare function chain<T0, T1 extends T0, T2 extends T1, T3 extends T2>(...tests: [JSONGuard<T1, T0>, JSONGuard<T2, T1>, JSONGuard<T3, T2>]): JSONGuard<T3, T0>;
export declare function chain<T0, T1 extends T0, T2 extends T1, T3 extends T2, T4 extends T3>(...tests: [
    JSONGuard<T1, T0>,
    JSONGuard<T2, T1>,
    JSONGuard<T3, T2>,
    JSONGuard<T4, T3>
]): JSONGuard<T4, T0>;
declare type MapType<Type> = {
    [Property in keyof Type]: Type[Property] extends (x: unknown) => infer U ? U : never;
};
export declare const object: <O extends Record<string, JSONGuard<unknown, unknown>>>(tests: O) => (x: unknown) => JSONGuardResult<MapType<O>>;
declare const _default: {
    boolean: JSONGuard<boolean, unknown>;
    number: JSONGuard<number, unknown>;
    string: JSONGuard<string, unknown>;
    constant: <T>(c: T) => JSONGuard<T, unknown>;
    email: JSONGuard<string, unknown>;
    chain: typeof chain;
};
export default _default;
