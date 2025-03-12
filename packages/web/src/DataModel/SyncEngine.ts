// import { FirebaseUser } from '../Pages/App/WithFirebaseUser'
// import { assert } from '../Utils/Utils'
// import { DataModel } from './DataModel'

// // TODO: Keep track of commited vs uncommited planParams, because we want to 
// // recalcualte CPB when that changes. Actually we are allowing paramsId to change, so 
// // maybe use that to track?
// export class SyncEngine {
//   private dataModel: DataModel.Main | null = null
//   constructor(
//     firebaseUser: FirebaseUser | null,
//     private readonly onChange: (dataModel: DataModel.Main) => void,
//     private readonly onError: (error: Error) => void,
//   ) {
//     DataModel.initialize(firebaseUser)
//       .then((dataModel) => {
//         this.dataModel = dataModel
//         this.onChange(dataModel)
//         return
//       })
//       .catch((error) => {
//         assert(error instanceof Error)
//         this.onError(error)
//       })
//   }

//   destroy() {
//     // TODO:
//   }

//   update(){

//   }

// }
