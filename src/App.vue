<template>
  <div id="app">
    <ImputationChart />
  </div>
</template>

<script>
import ImputationChart from './components/Training.vue';

export default {
  name: 'App',
  components: {
    ImputationChart
  }
};
</script>

<style>
/* 可选全局样式 */
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  margin: 20px;
}
</style>
