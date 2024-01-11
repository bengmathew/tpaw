import cloneJSONIn from 'fast-json-clone'

// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
export const cloneJSON = (cloneJSONIn as any).default as typeof cloneJSONIn
