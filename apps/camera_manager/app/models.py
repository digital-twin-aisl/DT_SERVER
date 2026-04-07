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
