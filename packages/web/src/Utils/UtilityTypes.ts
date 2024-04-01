export type PickType<T extends { type: string }, U extends T['type']> = Extract<
  T,
  { type: U }
>
