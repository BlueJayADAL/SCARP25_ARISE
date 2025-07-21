import React from 'react'
import { useSelector } from 'react-redux'

import {
  CSidebar,
  CSidebarFooter,
  CSidebarHeader
} from '@coreui/react'

import BluePose from 'src/assets/brand/bluePose'
import hookService from '../share/hookService'
import { AppSidebarNav } from './AppSidebarNav'
// sidebar nav config
import navigation from '../_nav'

const AppSidebar = () => {
  const footerMenuSelected = useSelector((state) => state.footerMenuSelected)
  const {isSidebarVisible} = hookService()  

  return (
    <CSidebar
      className="border-end"
      colorScheme="dark"
      position="fixed"
      unfoldable={false}
      visible={isSidebarVisible}
    >
      <CSidebarHeader className="border-bottom">
        <div className='custom-sidebar-brand'>
          <BluePose className="sidebar-brand-full" width={45} height={45} />
          <h4 className='sidebar-brand-narrow'>
            <span className="color-orange">ARISE</span>
            </h4>
        </div>
      </CSidebarHeader>
      <AppSidebarNav items={navigation[footerMenuSelected]} />
      <CSidebarFooter className="border-top d-none d-lg-flex">
        log in
        {/* <CSidebarToggler
          onClick={() => dispatch({ type: 'set', sidebarUnfoldable: !unfoldable })}
        /> */}
      </CSidebarFooter>
    </CSidebar>
  )
}

export default React.memo(AppSidebar)
