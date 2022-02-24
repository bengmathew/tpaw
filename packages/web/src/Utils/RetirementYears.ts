export const retirementYears = (
  params: {age: {start: number; retirement: number}},
  data: number[]
) => data.slice(params.age.retirement - params.age.start)
