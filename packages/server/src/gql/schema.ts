import { builder } from './builder.js'
import './GQLCommon/GQLSuccess.js'
import './GQLPlan/GQLPlan.js'
import './GQLLinkBasedPlan/GQLCreateLinkBasedPlan.js'
import './GQLLinkBasedPlan/GQLLinkBasedPlan.js'
import './GQLUser/GQLSendSignInEmail.js'
import './GQLUser/GQLSetUserPlan.js'
import './GQLUser/GQLUser.js'

export const schema = builder.toSchema()
