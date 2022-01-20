import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {AppPage} from '../App/AppPage'

export const TOS = React.memo(
  ({content, title}: {content: Contentful.FetchedInline; title: string}) => {
    return (
      <AppPage title={`${title} - TPAW Planner`}>
        <div className="">
          <Contentful.RichText
            body={content.fields.body}
            h1="font-bold text-4xl "
            h2="font-bold text-2xl mt-6"
            p="mt-4"
          />
        </div>
      </AppPage>
    )
  }
)
