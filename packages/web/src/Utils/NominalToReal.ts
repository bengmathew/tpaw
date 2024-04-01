export const nominalToReal = ({
  value: { amount, nominal },
  monthlyInflation,
  monthsFromNow,
}: {
  value: { amount: number; nominal: boolean }
  monthlyInflation: number
  monthsFromNow: number
}) => (nominal ? amount / Math.pow(1 + monthlyInflation, monthsFromNow) : amount)
