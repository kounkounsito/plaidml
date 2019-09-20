local PARAMS = {
  nvidia_test: {
    LOCAL_MEM_KIB: 48,
    NUM_THREADS: 256,
    CACHE_WIDTH: 128,
    NUM_UNITS: 15,
    REGS_MEM_B: 1024,
    REG_MEM_LAT: 1,
    LOCAL_MEM_LAT: 30,
    GLOBAL_MEM_LAT: 100,
    ALIGN_SIZE_B: 64
  },
};

{
  configs: {
    [cfg]: {
      stages: {
        default: {
          // Define the stripe passes
          passes: [
            // First, we place all the initial buffer in global memory (DRAM)
            {
              name: 'loc_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass',
                reqs: ['program'],
                loc: { devs: [{ name: 'GLOBAL', units: [{ offset: 0 }] }] },
              },
            },

            {
              name: 'loc_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass',
                reqs: ['main'],
                loc: { devs: [{ name: 'GLOBAL', units: [{ offset: 0 }] }] },
              },
            },

            // Prune indexes
            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['all'],
              },
            },

            // Eliminate the dead code first
            {
              name: 'dead_code_elimination',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.DeadCodeEliminationPass',
                reqs: ['all'],
              },
            },

            // Lower temps
            {
              name: 'localize_tmps',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['program'],
                ref_reqs: ['tmp'],
              },
            },

            // Pad tensors to remove inner conditionals
            {
              name: 'pad',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PadPass',
                reqs: ['main'],
              },
            },

            {
              name: 'generate_tiles',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.GenerateTilesPass',
                reqs: ['main'],
                outer_set: ['contract_outer', 'kernel'],
                inner_set: ['contract_inner'],
                acc_idxs: true,
                interleave: false,
                only_po2: true,
                odd_size: true,
                max_mem_size: PARAMS[cfg].LOCAL_MEM_KIB * 1024,
              }
            },

            {
              name: 'tile_middle',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.AutotilePass',
                reqs: ['contract_outer'],
                inner_set: ['contract_middle'],
                acc_idxs: false,
                input_cost: 0.0, 
                output_cost: 0.0,
                min_out_count: PARAMS[cfg].NUM_UNITS,
                split_factor: -100.0,
                only_po2: true,
              }
            },

            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['all'],
              },
            },

            {
              name: 'cache_input',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.CachePass',
                reqs: ['contract_middle'],
                ref: 'contract_inner',
                dirs: [ 'In' ],
                mem_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                xfer_loc: {},
                odd_size: true,
              }
            },

            {
              name: 'cache_output',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.CachePass',
                reqs: ['contract_outer'],
                ref: 'contract_inner',
                dirs: [ 'Out', 'InOut' ],
                mem_loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
                xfer_loc: {},
                odd_size: true,
              }
            },

            {
              name: 'prune_idxs',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['all'],
              },
            },

            {
              name: 'reduce_constraints',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.IlpConstraintReductionPass',
                reqs: ['all'],
              },
            },

            {
              name: 'localize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['main'],
              },
            },

            {
              name: 'scalarize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ScalarizePass',
                reqs: ['main'],
              },
            },

            ## We now relabel any remaining buffer inside the contractions as local memory
            {
              name: 'make_local',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateMemoryPass',
                reqs: ['contract_outer'],
                loc: { 'devs': [{'name': 'LOCAL', 'units': [{'offset': 0}]}] },
              }
            },

            {
              name: 'thread_cache',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['cache'],
                outer_set: ['cache_outer', 'gpu_thread'],
                inner_set: ['cache_threads', 'inline'],
                threads: PARAMS[cfg].NUM_THREADS,
              }
            },

            {
              name: 'thread_contract',
              pass : {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ThreadInnerPass',
                reqs: ['contract_inner'],
                outer_set: ['contract_inner', 'gpu_thread'],
                inner_set: ['contract_inner_threads', 'inline'],
                threads: PARAMS[cfg].NUM_THREADS,
              }
            },

            {
              name: 'loc_gpu',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocateInnerBlockPass',
                reqs: ['kernel'],
                loc: { 'devs': [{'name': 'GPU', 'units': [{'offset': 0}]}] }
              }
            },

            {
              name: 'cleanup1',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneRefinementsPass',
                reqs: ['main'],
              },
            },

            {
              name: 'cleanup2',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.PruneIndexesPass',
                reqs: ['main'],
              },
            },

            {
              name: 'localize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.LocalizePass',
                reqs: ['main'],
              },
            },

            {
              name: 'scalarize_main',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ScalarizePass',
                reqs: ['main'],
              },
            },

            {
              name: 'compute_deps',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.ComputeDepsPass', 
              }
            }, 

            {
              name: 'place_program',
              pass: {
                '@type': 'type.vertex.ai/vertexai.tile.codegen.proto.MemoryPlacementPass',
                reqs: ['program'],
                locs: [{ 'devs': [{'name': 'GLOBAL', 'units': [{'offset': 0}]}] }],
                alignment: 4,
              }
            }
          ],
        },
      },
    }
    for cfg in std.objectFields(PARAMS)
  },
}
