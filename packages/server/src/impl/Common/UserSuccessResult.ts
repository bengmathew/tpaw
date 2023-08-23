export type UserSuccessResult = { type: 'UserSuccessResult'; userId: string }
export const getUserSuccessResult = (userId: string): UserSuccessResult => ({
  type: 'UserSuccessResult',
  userId,
})
