// Adds the undefined type.
export const optGet = <T>(
  x: Record<string, T>,
  key: string,
): T | undefined => x[key]
