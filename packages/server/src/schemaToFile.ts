import {writeFileSync} from 'fs'
import {lexicographicSortSchema, printSchema} from 'graphql'
import path from 'path'
import {schema} from './gql/schema'

export function schemaToFile() {
  writeFileSync(
    path.join(process.cwd(), '/generated/schema.graphql'),
    printSchema(lexicographicSortSchema(schema)),
  )
}

schemaToFile()
