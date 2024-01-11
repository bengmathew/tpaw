import { File } from '@google-cloud/storage'
import { API, assert } from '@tpaw/common'
import _ from 'lodash'
import * as uuid from 'uuid'
import { Clients } from '../../Clients.js'
import { Config } from '../../Config.js'
import { builder } from '../builder.js'

const Input = builder.inputType('GeneratePDFReportInput', {
  fields: (t) => ({
    url: t.string(),
    auth: t.string({ required: false }),
    viewportWidth: t.int(),
    viewportHeight: t.int(),
    devicePixelRatio: t.int(),
  }),
})

const Output = builder
  .objectRef<{ type: 'GeneratePDFReportOutput'; pdfURL: string }>(
    'GeneratePDFReportOutput',
  )
  .implement({
    fields: (t) => ({
      pdfURL: t.string({ resolve: (value) => value.pdfURL }),
    }),
  })

builder.mutationField('generatePDFReport', (t) =>
  t.field({
    type: Output,
    args: { input: t.arg({ type: Input }) },
    resolve: async (__, { input }) => ({
      type: 'GeneratePDFReportOutput' as const,
      pdfURL: await generatePDFReport(
        API.GeneratePDFReport.check(input).force(),
      ),
    }),
  }),
)

export const generatePDFReport = async (input: API.GeneratePDFReport.Input) => {
  const sStart = Date.now()
  const pdfBuffer = await _getPDFBufferFromPrintServer(input)

  const bucket = Clients.gcs.bucket(Config.google.transientDataBucket)
  const file = new File(bucket, `${uuid.v4()}.pdf`)
  let start = Date.now()
  await file.save(pdfBuffer, {
    contentType: 'application/pdf',
    public: false,
  })
  console.log('browser.saveFile', Date.now() - start)
  start = Date.now()
  const [pdfURL] = await file.getSignedUrl({
    action: 'read',
    expires: Date.now() + 1000 * 60 * 60 * 2,
  })
  console.log('browser.getSignedUrl', Date.now() - start)
  console.log('TOTAL', Date.now() - sStart)
  return pdfURL
}

const _getPDFBufferFromPrintServer = async ({
  url,
  auth,
  viewportWidth,
  viewportHeight,
  devicePixelRatio,
}: API.GeneratePDFReport.Input) => {
  type Args = {
    url: string
    auth: string | null
    viewportWidth: number
    viewportHeight: number
    devicePixelRatio: number
  }
  const args: Args = {
    url,
    auth,
    viewportWidth,
    viewportHeight,
    devicePixelRatio,
  }
  const response = await fetch(new URL(`/print`, Config.printServer.url), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: _.compact([
        Config.debug.authHeader ?? null,
        `Bearer ${Config.interServerToken}`,
      ]).join(', '),
    },
    body: JSON.stringify(args),
  })
  assert(response.ok)
  return Buffer.from(await (await response.blob()).arrayBuffer())
}
