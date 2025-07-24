import { CIcon } from '@coreui/icons-react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom'
import { AppContent, AppHeader, AppSidebar } from '../components/index';
import { footerMenu } from '../share/constants';
import hookService from '../share/hookService';
import HandCursor from '../components/virtualMouse/handCursor';
import useHandTracking from '../components/virtualMouse/useHandTracking';

const DefaultLayout = () => {

  const dispatch = useDispatch()
  const footerMenuSelected = useSelector((state) => state.footerMenuSelected)
  const navigate = useNavigate();

  const updateFooterMenuSelected = (footerMenu) => {
    dispatch({ type: 'set', footerMenuSelected: footerMenu })
    dispatch({ type: 'set', sidebarShow: true })
    navigate("/" + footerMenu)
  }

  const { position, indexClicking } = useHandTracking();

  const {isSidebarVisible} = hookService()

  const footerMenuItems = Object.keys(footerMenu).map((key) => {
    const item = footerMenu[key]
    return (
      <div
       key={key} className={footerMenuSelected === item.link ? 'blue-adal-footer-item active' : 'blue-adal-footer-item'} 
       onClick={() => updateFooterMenuSelected(item.link)}>
        <h4>{item.name}</h4>
        <CIcon icon={item.icon} width={50} height={50} />
      </div>
    )
  })

  return (
    <div className='blue-adal-content'>
      <HandCursor position={position} isLeftClicking={indexClicking} />
      <AppSidebar />
      <div className="wrapper d-flex flex-column max-vh-100">
        <AppHeader />
         <div className="body flex-grow-1">
          <AppContent />
        </div>
      </div>
      <div  className={isSidebarVisible ? "blue-adal-footer open" : "blue-adal-footer close"}>
        {footerMenuItems}
      </div>
    </div>
  )
}

export default DefaultLayout
