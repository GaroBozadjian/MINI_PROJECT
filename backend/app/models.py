from sqlalchemy import Integer, Float
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base

class IrisRow(Base):
    __tablename__ = "iris"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sepal_length: Mapped[float] = mapped_column(Float, nullable=False)
    sepal_width: Mapped[float] = mapped_column(Float, nullable=False)
    petal_length: Mapped[float] = mapped_column(Float, nullable=False)
    petal_width: Mapped[float] = mapped_column(Float, nullable=False)

    # store target too (0/1/2) so you can analyze by class
    target: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
