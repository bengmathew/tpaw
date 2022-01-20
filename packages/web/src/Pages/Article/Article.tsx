import {GetStaticProps, InferGetStaticPropsType} from 'next'
import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {assert, fGet} from '../../Utils/Utils'
import {AppPage} from '../App/AppPage'

export const articleGetStaticProps: GetStaticProps<{
  content: Contentful.FetchedInline
}> = async ({params}) => {
  const slug = fGet(params?.slug)
  assert(typeof slug === 'string')
  return {
    props: {
      content: await Contentful.fetchStandalone(slug),
    },
  }
}

export const Article = React.memo(
  ({content}: InferGetStaticPropsType<typeof articleGetStaticProps>) => {
    return (
      <AppPage title={`${content.fields.title} - TPAW Planner`}>
        <div className="flex justify-center">
          <div className="max-w-[800px]">
            <h1 className="text-3xl font-bold mb-6">{content.fields.title}</h1>
            <Contentful.RichText
              body={content.fields.body}
              p="mb-2"
              h2="font-bold text-2xl"
            />
          </div>
        </div>
      </AppPage>
    )
  }
)
