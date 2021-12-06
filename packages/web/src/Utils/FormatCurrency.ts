// creating the formatter on server was causing "maximumFractionDigits value is out of range"
export const formatCurrency = (x: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(x)
