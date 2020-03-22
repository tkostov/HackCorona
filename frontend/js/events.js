let app = new Vue({
  el: '.container-fluid',
  data: {
    name: 'Vue.js'
  },
  // define methods under the `methods` object
  methods: {
    heatmapFunctionCases: function (event) {
      cleanHeatmap();
      heatmapFunctionCases();
    },
    heatmapFunctionDeads: function (event) {
      cleanHeatmap();
      heatmapFunctionDeads();
    }
  }
})