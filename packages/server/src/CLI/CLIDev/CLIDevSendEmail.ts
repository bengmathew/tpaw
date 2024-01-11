import fs from 'fs-extra'
import { Clients } from '../../Clients.js'
import { cliDev } from './CLIDev.js'

cliDev
  .command('sendEmail <email> <messageStream> <subject> <messageFile>')
  .action(
    async (
      email: string,
      messageStream: string,
      subject: string,
      messageFile: string,
    ) => {
      const message = fs.readFileSync(messageFile).toString()
      await Clients.postmark.sendEmail({
        From: 'hello@tpawplanner.com',
        To: email,
        Subject: subject,
        MessageStream: messageStream,
        HtmlBody: message,
      })
    },
  )
