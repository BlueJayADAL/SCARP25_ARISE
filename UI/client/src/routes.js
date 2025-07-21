import React from 'react'

const Dashboard = React.lazy(() => import('./views/dashboard/Dashboard'))

// Exercices
const Curls = React.lazy(() => import('./views/exercices/Curls'))
const Bicep = React.lazy(() => import('./views/exercices/Bicep'))


// Games
const SpaceInvaders = React.lazy(() => import('./views/games/SpaceInvaders'))
const FalppyBird = React.lazy(() => import('./views/games/FalppyBird'))
const FruitNinja = React.lazy(() => import('./views/games/FruitNinja'))


// Games
const Chat = React.lazy(() => import('./views/chat/Chat'))


const routes = [
  { path: '/', exact: true, name: 'Home' },
  { path: '/dashboard', name: 'Dashboard', element: Dashboard },
  { path: '/exercises', name: 'Bicep', element: Bicep, exact: true },
  { path: '/exercises/bicep', name: 'Bicep', element: Bicep, exact: true },
  { path: '/exercises/curls', name: 'Curls', element: Curls },
  { path: '/games', name: 'Space Invaders', element: SpaceInvaders, exact: true },
  { path: '/games/space-invaders', name: 'Space Invaders', element: SpaceInvaders, exact: true },
  { path: '/games/flappy-bird', name: 'Falppy Bird', element: FalppyBird },
  { path: '/games/fruit-ninja', name: 'Fruit Ninja', element: FruitNinja },
  { path: '/chat', name: 'Chat', element: Chat, exact: true },
]

export default routes
