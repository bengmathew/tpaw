export const nominalToReal = ({
  value: { value, nominal },
  monthlyInflation,
  monthsFromNow,
}: {
  value: { value: number; nominal: boolean }
  monthlyInflation: number
  monthsFromNow: number
}) => (nominal ? value / Math.pow(1 + monthlyInflation, monthsFromNow) : value)
