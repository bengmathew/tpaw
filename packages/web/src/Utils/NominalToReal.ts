    
export const nominalToReal = (
  {value, nominal}: {value: number; nominal: boolean},
  inflation:number,
  yearsFromNow: number
) => (nominal ? value / Math.pow(1 + inflation, yearsFromNow) : value)
