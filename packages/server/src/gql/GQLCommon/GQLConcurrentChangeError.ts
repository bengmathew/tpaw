import {builder} from '../builder.js'

builder.objectType('ConcurrentChangeError', {
  fields: t => ({
    _: t.int({resolve: () => 0}),
  }),
})
