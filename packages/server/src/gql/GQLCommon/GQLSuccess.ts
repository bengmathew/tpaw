import {builder} from '../builder.js'

builder.objectType('Success', {
  fields: t => ({
    _: t.int({resolve: () => 0}),
  }),
})
