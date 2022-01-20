import _ from 'lodash'
import {GetStaticProps, InferGetStaticPropsType} from 'next'
import Head from 'next/head'
import Link from 'next/link'
import React from 'react'
import {Contentful} from '../../Utils/Contentful'
import {assert} from '../../Utils/Utils'

// Contents of pages/knowledge-base/[[...slug]].tsx
// import {
//   KnowledgeBase,
//   knowledgeBaseGetStaticProps,
// } from '../../src/Pages/KnowledgeBase/KnowledgeBase'
// import {Contentful} from '../../src/Utils/Contentful'

// export default KnowledgeBase

// export const getStaticProps = knowledgeBaseGetStaticProps

// export async function getStaticPaths() {
//   const slugs = await Contentful.getKnowledgeBaseArticleSlugs()

//   return {
//     paths: slugs.map(slug => ({
//       params: {slug: slug === 'knowledge-base-start' ? [] : [slug]},
//     })),
//     fallback: false,
//   }
// }


export const knowledgeBaseGetStaticProps: GetStaticProps<{
  outline: Contentful.FetchedKnowledgeBaseOutline
  content: Contentful.FetchedKnowledgeBaseArticle
}> = async ({params}) => {
  const slugArr = params?.slug ?? []
  assert(slugArr instanceof Array)
  if (slugArr.length > 1) return {notFound: true}
  const slug = slugArr.length > 0 ? slugArr[0] : 'knowledge-base-start' // ignore everything but last
  return {
    props: {
      outline: await Contentful.fetchKnowledgeBaseOutline(
        '4QWP7hVZ4lZ3ZD8Bz1nWH3'
      ),
      content: await Contentful.fetchKnowledgeBaseArticle(slug),
    },
  }
}

export const KnowledgeBase = React.memo(
  ({
    outline,
    content,
  }: InferGetStaticPropsType<typeof knowledgeBaseGetStaticProps>) => {
    return (
      <div
        className="font-font1 text-gray-800 grid gap-x-5  min-h-screen"
        style={{grid: '1fr/ auto 1fr'}}
      >
        <Head>
          <title>
            {content.fields.publicTitle} - Knowledge Base - TPAW Planner
          </title>
        </Head>
        <div className="bg-gray-100 min-w-[300px] h-full p-2">
          <_Outline className="" outline={outline} />
        </div>
        <div className="p-2">
          <h1 className="text-4xl font-bold mb-6">
            {content.fields.publicTitle}
          </h1>
          <Contentful.RichText
            body={content.fields.body}
            p="mb-2"
            h2="font-bold text-2xl"
          />
        </div>
      </div>
    )
  }
)

const _Outline = React.memo(
  ({
    className = '',
    outline,
  }: {
    className?: string
    outline: Contentful.FetchedKnowledgeBaseOutline
  }) => {
    return (
      <div className={`${className}`}>
        <div className="ml-4">
          {outline.fields.items.map((item, i) => {
            console.dir(item)
            if (_isArticle(item)) {
              const {slug, publicTitle} = item.fields
              return <_Link key={i} slug={slug} title={publicTitle} />
            } else {
              return (
                <div className="">
                  <_Link
                    key={i}
                    className="font-semibold py-2"
                    slug={_leaves(item)[0].fields.slug}
                    title={item.fields.title}
                  />
                  <_Outline key={i} className="" outline={item} />
                </div>
              )
            }
          })}
        </div>
      </div>
    )
  }
)

const _Link = React.memo(
  ({
    className = '',
    slug,
    title,
  }: {
    className?: string
    slug: string
    title: string
  }) => (
    <Link href={`/knowledge-base/${slug === 'knowledge-base-start' ? '' : slug}`}>
      <a className="block font-semibold  py-2"> {title}</a>
    </Link>
  )
)

const _leaves = (
  outline: Contentful.FetchedKnowledgeBaseOutline
): Contentful.FetchedKnowledgeBaseArticle[] => {
  return _.flatten(
    outline.fields.items.map(item =>
      _isArticle(item) ? [item] : _leaves(item)
    )
  )
}

const _isArticle = (
  item:
    | Contentful.FetchedKnowledgeBaseOutline
    | Contentful.FetchedKnowledgeBaseArticle
): item is Contentful.FetchedKnowledgeBaseArticle =>
  'publicTitle' in item.fields
