import { builder } from '../builder.js'

export const SyncClientInput = builder.inputType('SyncClientInput', {
  fields: (t) => ({
    id: t.string(),
    createdAt: t.float(),
  }),
})
