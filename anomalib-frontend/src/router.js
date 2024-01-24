import { createRouter, createWebHistory } from "vue-router";
import AnomalyTraining from "./components/AnomalyTraining.vue";
import AnomalyPrediction from "./components/AnomalyPrediction.vue";

const routes = [
  { path: "/training", component: AnomalyTraining },
  { path: "/prediction", component: AnomalyPrediction },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
