import jsonpath from 'fast-json-patch'
import { performance } from 'perf_hooks'
import { cli } from './CLI.js'
import { cloneJSON } from '../Utils/CloneJSON.js'
import { getDefaultNonPlanParams, nonPlanParamsGuard } from '@tpaw/common'

cli.command('scratch ').action(async () => {

    const nonPlanParams = getDefaultNonPlanParams()
    console.dir(nonPlanParams)
    console.dir(nonPlanParamsGuard(nonPlanParams))
})
