declare type MapType<Type> = {
    [Property in keyof Type]: Type[Property] extends (x: unknown) => infer U ? U : never;
};
export declare type Validator<T, F = unknown> = (x: F) => T;
export declare namespace Validator {
    class Failed extends Error {
        path: string | undefined;
        lines: string[];
        constructor(lines: string | string[], path?: string);
        get fullLines(): string[];
        get fullMessage(): string;
    }
    const number: () => (x: unknown) => number;
    const boolean: () => (x: unknown) => boolean;
    const string: () => (x: unknown) => string;
    const constant: <C extends string | number | boolean | null>(c: C) => (x: unknown) => C;
    const array: <T>(test: Validator<T, unknown>) => (x: unknown) => T[];
    const object: <O extends Record<string, Validator<any, unknown>>>(tests: O) => (x: unknown) => MapType<O>;
    const union: <T extends Validator<any, unknown>[]>(...tests: T) => Validator<MapType<T>[number], unknown>;
    function intersection<T1, T2>(...tests: [Validator<T1>, Validator<T2>]): Validator<T1 & T2>;
    function intersection<T1, T2, T3>(...tests: [Validator<T1>, Validator<T2>, Validator<T3>]): Validator<T1 & T2 & T3>;
    function chain<T0, T1, T2>(...tests: [Validator<T1, T0>, Validator<T2, T1>]): Validator<T2, T0>;
    function chain<T0, T1, T2, T3>(...tests: [Validator<T1, T0>, Validator<T2, T1>, Validator<T3, T2>]): Validator<T3, T0>;
    function chain<T0, T1, T2, T3>(...tests: [Validator<T1, T0>, Validator<T2, T1>, Validator<T3, T2>]): Validator<T3, T0>;
    function chain<T0, T1, T2, T3, T4>(...tests: [
        Validator<T1, T0>,
        Validator<T2, T1>,
        Validator<T3, T2>,
        Validator<T4, T3>
    ]): Validator<T4, T0>;
}
export {};
