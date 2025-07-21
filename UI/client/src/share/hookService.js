import React, { useMemo } from 'react'
import { useSelector } from 'react-redux'
const hookService = () => {

    const sidebarShow = useSelector((state) => state.sidebarShow)
    const footerMenuSelected = useSelector((state) => state.footerMenuSelected)

    const isSidebarVisible = useMemo (() => {
        return sidebarShow && footerMenuSelected !== 'dashboard' && footerMenuSelected !== 'chat'
    }, [sidebarShow, footerMenuSelected])


    return {
        isSidebarVisible
    }
}

export default hookService