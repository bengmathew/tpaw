import {GetStaticProps} from 'next'
import {TOS} from '../src/Pages/TOS/TOS'
import {Contentful} from '../src/Utils/Contentful'

export const getStaticProps: GetStaticProps<{
  content: Contentful.FetchedInline
}> = async () => ({
  props: {content: await Contentful.fetchInline('6UU02icvCcZdMlbVTjRzlD'), title:'About'},
})
export default TOS
