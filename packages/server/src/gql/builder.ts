import SchemaBuilder from '@pothos/core'
import PrismaPlugin from '@pothos/plugin-prisma'
import type PrismaTypes from '@pothos/plugin-prisma/generated'
import ScopeAuthPlugin from '@pothos/plugin-scope-auth'
import { Clients } from '../Clients.js'
import { ConcurrentChangeError } from '../impl/Common/ConcurrentChangeError.js'
import { Success } from '../impl/Common/Success.js'
import { UserSuccessResult } from '../impl/Common/UserSuccessResult.js'
import { Context } from './Context.js'
import { PlanParamsChangePatched } from './GQLUser/GQLUserPlan/PatchPlanParams.js'

export const builder = new SchemaBuilder<{
  Context: Context
  DefaultInputFieldRequiredness: true
  Objects: {
    Success: Success
    ConcurrentChangeError: ConcurrentChangeError
    UserSuccessResult: UserSuccessResult
    PlanParamsChangePatched: PlanParamsChangePatched
  }
  PrismaTypes: PrismaTypes
  AuthScopes: {
    user: boolean
    admin: boolean
  }
}>({
  defaultInputFieldRequiredness: true,
  plugins: [ScopeAuthPlugin, PrismaPlugin],
  authScopes: (context) => ({
    user: context.user !== null,
    admin: false,
  }),
  prisma: {
    client: Clients.prisma,
    filterConnectionTotalCount: true,
  },
})

builder.queryType()
builder.mutationType()
builder.queryField('ping', (t) => t.string({ resolve: () => 'pong' }))
builder.queryField('crash', (t) =>
  t.string({
    resolve: () => {
      throw new Error('Crash!')
    },
  }),
)
builder.mutationField('crash', (t) =>
  t.field({
    type: 'Success',
    resolve: async () => {
      throw new Error('Crash!')
    },
  }),
)
