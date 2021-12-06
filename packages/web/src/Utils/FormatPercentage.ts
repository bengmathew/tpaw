// creating the formatter on server was causing "maximumFractionDigits value is out of range"
export const formatPercentage =(precision:number)=> (x: number) =>
  `${(x * 100).toFixed(precision)}%`
