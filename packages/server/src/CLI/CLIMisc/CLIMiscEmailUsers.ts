import { assert, fGet } from '@tpaw/common'
import fs from 'fs-extra'
import inquirer from 'inquirer'
import _ from 'lodash'
import { Message } from 'postmark'
import { Clients } from '../../Clients.js'
import { cliMisc } from './CLIMisc.js'

cliMisc
  .command('emailUsers <messageStream> <subject> <messageFile>')
  .action(
    async (messageStream: string, subject: string, messageFile: string) => {
      assert(messageStream === 'compliance')
      const message = fs.readFileSync(messageFile).toString()
      //500 is batch limit.
      const usersUnchunked = await Clients.prisma.user.findMany({})

      const { shouldContinue } = (await inquirer.prompt({
        type: 'confirm',
        name: 'shouldContinue',
        message: `Are you sure you want to send this email to  ${usersUnchunked.length} users?`,
        default: false,
      })) as unknown as { shouldContinue: boolean }
      if (!shouldContinue) return

      for (const usersChunk of _.chunk(usersUnchunked, 250)) {
        const firebaseUsers = await Promise.all(
          usersChunk.map(
            async ({ id }) => await Clients.firebaseAuth.getUser(id),
          ),
        )

        const mesages: Message[] = usersChunk.map((user, i) => ({
          From: 'hello@tpawplanner.com',
          To: fGet(fGet(firebaseUsers[i]).email),
          Subject: subject,
          MessageStream: messageStream,
          HtmlBody: message,
        }))
        await Clients.postmark.sendEmailBatch(mesages)
      }
    },
  )
