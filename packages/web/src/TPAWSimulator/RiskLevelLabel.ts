import { noCase } from '../Utils/Utils'
import { TPAWRiskLevel } from './TPAWParams'

export const riskLevelLabel = (riskLevel:TPAWRiskLevel)=>{
  switch(riskLevel){
    case 'riskLevel-1': return 'Conservative'
    case 'riskLevel-2': return 'Moderately Conservative'
    case 'riskLevel-3': return 'Moderately Aggressive'
    case 'riskLevel-4': return 'Aggressive'
    default:noCase(riskLevel)
  }
}