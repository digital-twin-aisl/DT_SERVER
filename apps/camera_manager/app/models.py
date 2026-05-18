# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class EdgeDevice(Base):
    __tablename__ = "edge_devices"

    id = Column(String, primary_key=True, index=True) # ex: edge-jetson-01
    ip_address = Column(String)
    location = Column(String)
    
    cameras = relationship("Camera", back_populates="edge")

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(String, ForeignKey("edge_devices.id"))
    onvif_url = Column(String)  # 카메라의 ONVIF 스트림 URL 또는 IP 정보
    name = Column(String)
    
    # 카메라 캘리브레이션 메타데이터 (내외부 파라미터 등)
    intrinsic_params = Column(JSON, nullable=True)
    extrinsic_params = Column(JSON, nullable=True)

    edge = relationship("EdgeDevice", back_populates="cameras")
