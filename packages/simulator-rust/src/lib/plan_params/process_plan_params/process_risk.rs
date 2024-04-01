use serde::{Deserialize, Serialize};
use tsify::Tsify;

use crate::{
    plan_params_rust::{GlidePath},
    shared_types::{MonthAndStocks, StocksAndBonds},
};

// fn process_glide_path(glide_path: GlidePath, num_simulation_months: i64) {
//     // Keeps only the first  of duplicates and prioritizes "start" and "end", over
//     // intermediate if they refer to same month.
//     let stage1 = {
//         let mut flat = [
//             vec![
//                 glide_path.start,
//                 MonthAndStocks {
//                     month: num_simulation_months - 1,
//                     stocks: glide_path.end.stocks,
//                 },
//             ],
//             {
//                 let mut values = glide_path
//                     .intermediate
//                     .iter()
//                     .map(|(k, v)| v)
//                     .collect::<Vec<_>>();
//                 values.sort_by_key(|x| x.index_to_sort_by_added);
//                 values
//                     .iter()
//                     .map(|x| MonthAndStocks {
//                         month: x.month,
//                         stocks: x.stocks,
//                     })
//                     .collect::<Vec<_>>()
//             },
//         ]
//         .concat();
//         flat.sort_by_key(|x| x.month);
//         flat.dedup_by_key(|x| x.month);
//         flat
//     };
// }
